"""
06_path_planning.py
04_bev_traversability.py 가 만드는 BEV traversability cost map 위에서, GeNIE 논문
(Sec III-D, Algorithm 1)의 "후보 경로 샘플링 -> top-K 선별 -> path fusion(클러스터링
+ 근접 클러스터 병합) -> 목표 방향 정렬 기반 선택" 파이프라인을 재현한다. 결과로
[입력 이미지 | traversability | BEV | BEV+경로(후보=흰색, 선택=마젠타)] 4분할
이미지를 debug/ride_<id>/frame_<번호>_path.png 로 저장한다.

════════════════════════════════════════════════════════════════════════════
 이 구현이 논문/실제 ERC 대회 세팅과 다른 점 (중요, 읽고 쓰기)
════════════════════════════════════════════════════════════════════════════
1) GPS 목표(goal)가 없다.
   논문 Algorithm 1은 "GPS goal g"를 외부 입력으로 받는다 — 대회에서는 다음
   체크포인트의 GPS 좌표가 미리 주어진다. 반면 이 FrodoBots 주행 로그에는
   "미션 목표 좌표"라는 개념 자체가 없고, 그냥 사람(또는 기존 정책)이 실제로
   주행한 기록만 있다. 그래서 여기서는 "그 시점 이후 실제 GPS가 어디로
   이동했는가"를 목표의 대용(proxy)으로 쓴다: 현재 프레임 직전의 GPS 이동
   방향을 현재 heading으로, 몇 초 뒤(--goal-lookahead-s) GPS 위치까지의 방향을
   목표 방향으로 삼는다 (estimate_goal_angle_deg 참고). 이건 "미리 정해진
   목표를 향해 가는" 논문의 상황이 아니라 "실제로 갔던 곳을 사후에 목표라고
   가정"하는 것이므로, 이 스크립트의 결과는 "이 알고리즘이 실제 주행 방향과
   얼마나 비슷한 경로를 골랐는가"를 정성적으로 보는 용도로 이해해야 한다.
   --goal-angle-deg 로 직접 각도를 지정해 이 추정을 완전히 건너뛸 수도 있다.
2) 카메라-지면 캘리브레이션이 없다 (04번 스크립트와 동일한 한계).
   hfov/cam-height는 실측이 아니라 근사 파라미터다. 경로 샘플링의 각도 범위도
   이 근사된 hfov를 그대로 쓴다.
3) 실제 SAM-TP 가중치가 없다 (04/05번 스크립트와 동일한 한계) — traversability는
   zero-shot SAM2(point prompt) 또는 색상/텍스처 휴리스틱으로 근사한다.
4) 클러스터링/실루엣 계산을 sklearn 없이 numpy로 직접 구현했다. 논문이 실제
   내부적으로 sklearn을 썼는지는 코드가 공개돼 있지 않아 확인 불가능하지만,
   수식(Sec III-D, 식 2)은 표준 실루엣 계수라 결과는 동일한 원리로 동작한다.
5) path fusion의 merge threshold(--merge-threshold), top-K(--top-k) 등 정확한
   하이퍼파라미터 값은 논문에 PR curve(Fig.6a)로만 나오고 최종 채택값은 명시돼
   있지 않다. 여기서는 "일단 웬만하면 안 합친다(false negative를 선호)"는
   논문의 설계 원칙만 유지하고, 구체적인 값은 이 데이터셋에서 보기 좋게 나오는
   선에서 임의로 정했다 — 실전 값이 아니라 데모/테스트용 기본값임을 유의.
════════════════════════════════════════════════════════════════════════════

사용:
  python3 scripts/06_path_planning.py \
      data/minirover-0011/ride_104913_dgp79m_20250228025658 --num-samples 6

필요 패키지: numpy, opencv-python(-headless)  (+ --estimator sam2 쓰려면 torch, transformers)
"""
import os
import sys
import csv
import argparse
import importlib.util

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEBUG_DIR = os.path.join(REPO_ROOT, "debug")


def _load_bev_module():
    """04_bev_traversability.py는 파일명이 숫자로 시작해서 일반 `import` 문으로 못
    불러오므로, 파일 경로 기준으로 importlib로 직접 로드해서 재사용한다."""
    path = os.path.join(SCRIPT_DIR, "04_bev_traversability.py")
    spec = importlib.util.spec_from_file_location("bev_traversability", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bevmod = _load_bev_module()


# --------------------------------------------------------------------------
# 1) GPS 기반 "목표 방향" 추정 (논문의 GPS goal을 대신하는 이 데이터셋 전용 근사)
# --------------------------------------------------------------------------

def load_camera_timestamps(session_dir, rid):
    path = os.path.join(session_dir, f"front_camera_timestamps_{rid}.csv")
    frame_ids, ts = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            frame_ids.append(int(row["frame_id"]))
            ts.append(float(row["timestamp"]))
    return np.asarray(frame_ids), np.asarray(ts)


def load_gps(session_dir, rid):
    """gps_data_<rid>.csv 로딩. timestamp 컬럼은 다른 csv들(초 단위)과 달리
    밀리초 단위(예: 1740711418979)라 여기서 /1000 으로 맞춰준다."""
    path = os.path.join(session_dir, f"gps_data_{rid}.csv")
    if not os.path.isfile(path):
        return None
    ts, lat, lon = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts.append(float(row["timestamp"]) / 1000.0)
            lat.append(float(row["latitude"]))
            lon.append(float(row["longitude"]))
    return np.asarray(ts), np.asarray(lat), np.asarray(lon)


def _gps_to_local_xy(lat, lon, lat0, lon0):
    """위경도 -> (lat0, lon0)를 원점으로 하는 로컬 평면(East, North)[m] 근사 변환.
    이동 거리가 짧으므로(수십~수백 m) equirectangular 근사로 충분하다."""
    east = (lon - lon0) * np.cos(np.radians(lat0)) * 111320.0
    north = (lat - lat0) * 110540.0
    return east, north


def estimate_goal_angle_deg(gps, cam_ts, lookahead_s=8.0, min_move_m=0.3):
    """GPS 궤적에서 "논문의 GPS goal과 비슷한 효과"를 내는 목표 방향[deg]을 추정.

    - heading(현재 진행 방향): 현재 시각 직전 GPS 두 점을 이은 방향.
    - goal(목표 방향): 현재 시각으로부터 lookahead_s초 뒤 GPS 위치까지의 방향
      (= "이 로버가 실제로 향했던 곳"을 사후적으로 목표라고 가정).
    - 반환값은 이 두 방향의 차이(로봇 기준 상대각, +면 오른쪽)이다. 로봇이 정지해
      있어 방향을 신뢰할 수 없거나(min_move_m 미만 이동) GPS가 너무 성기면
      None을 반환하고, 호출부에서 0도(직진)로 대체한다.
    """
    if gps is None or cam_ts is None:
        return None
    gps_ts, gps_lat, gps_lon = gps
    if len(gps_ts) < 3:
        return None

    i_cur = int(np.argmin(np.abs(gps_ts - cam_ts)))
    i_prev = max(0, i_cur - 1)
    i_future = int(np.argmin(np.abs(gps_ts - (cam_ts + lookahead_s))))
    if i_future <= i_cur:
        i_future = min(len(gps_ts) - 1, i_cur + 1)
    if i_future == i_cur or i_prev == i_cur:
        return None

    lat0, lon0 = gps_lat[i_cur], gps_lon[i_cur]
    e_prev, n_prev = _gps_to_local_xy(gps_lat[i_prev], gps_lon[i_prev], lat0, lon0)
    e_fut, n_fut = _gps_to_local_xy(gps_lat[i_future], gps_lon[i_future], lat0, lon0)

    # heading 방향 벡터 = (현재 위치) - (직전 위치) = (0,0) - (e_prev, n_prev)
    if (e_prev ** 2 + n_prev ** 2) ** 0.5 < min_move_m:
        heading_bearing = 0.0  # 최근 이동이 거의 없으면 heading을 신뢰할 수 없음 -> 카메라 정면(0도)으로 가정
    else:
        heading_bearing = np.degrees(np.arctan2(-e_prev, -n_prev))

    if (e_fut ** 2 + n_fut ** 2) ** 0.5 < min_move_m:
        return None  # 미래 위치도 현재와 거의 같으면(정차 등) 목표 방향을 정의할 수 없음

    goal_bearing = np.degrees(np.arctan2(e_fut, n_fut))
    rel = ((goal_bearing - heading_bearing + 180.0) % 360.0) - 180.0
    return float(rel)


# --------------------------------------------------------------------------
# 2) 후보 경로 샘플링 + BEV 기반 비용 평가
# --------------------------------------------------------------------------

def sample_candidate_paths(hfov_deg, bev_range_m, n_paths=25, n_wp=10, end_frac=0.85,
                            curvature_jitter=0.2, seed=0):
    """로봇 원점(0,0)에서 시작해, 카메라 시야각(hfov) 안의 다양한 방향으로 뻗는
    경로 후보 M개를 샘플링한다 (논문: "parameterized as first- and second-order
    polynomials connecting start and end points"). 끝점은 반지름
    end_frac*bev_range_m인 호(arc) 위에 hfov 범위로 고르게 배치하고, 각 경로는
    t^curve_power 형태의 곡선으로 원점과 끝점을 연결한다 (초반엔 거의 직진하다가
    끝에서 목표 방향으로 꺾이는 형태 -> 로봇이 갑자기 옆으로 튀지 않는 자연스러운
    curvature). curvature_jitter로 경로마다 곡률에 약간의 무작위성을 줘서 다양한
    모양을 만든다 (논문 Fig.1/5의 부채꼴 모양 후보 경로들과 같은 취지).

    반환: (n_paths, n_wp, 2) — 각 waypoint는 (x=우측+, z=전방+)[m].
    """
    rng = np.random.default_rng(seed)
    radius = bev_range_m * end_frac
    angles_deg = np.linspace(-hfov_deg / 2, hfov_deg / 2, n_paths)
    t = np.linspace(0.0, 1.0, n_wp)

    paths = np.zeros((n_paths, n_wp, 2), dtype=np.float32)
    for i, ang in enumerate(angles_deg):
        theta = np.radians(ang)
        end_x = radius * np.sin(theta)
        end_z = radius * np.cos(theta)
        curve_power = 1.6 + rng.uniform(-curvature_jitter, curvature_jitter)
        paths[i, :, 0] = end_x * (t ** curve_power)
        paths[i, :, 1] = end_z * t
    return paths


def score_path_on_bev(path_xy, bev_score, known_mask, bev_range_m, unknown_value=0.15):
    """경로가 지나는 지점들의 BEV traversability 평균(논문 식1의 f(pi) 항에 대응,
    값이 클수록 traversability가 높다 = cost가 낮다). 아직 관측 안 된(known_mask
    False) 지점은 "잠재적으로 위험하다"고 보수적으로 unknown_value를 부여하고,
    BEV 범위를 벗어나는 지점은 0점(가지 말아야 할 곳)으로 취급한다."""
    grid_h, grid_w = bev_score.shape
    res = bev_range_m / grid_h
    xs, zs = path_xy[:, 0], path_xy[:, 1]

    out_of_range = (np.abs(xs) > bev_range_m / 2) | (zs < 0) | (zs > bev_range_m)
    col = np.clip(((xs + bev_range_m / 2) / res).astype(np.int32), 0, grid_w - 1)
    row = np.clip((grid_h - 1 - (zs / res).astype(np.int32)), 0, grid_h - 1)

    vals = np.where(known_mask[row, col], bev_score[row, col], unknown_value)
    vals = np.where(out_of_range, 0.0, vals)
    return float(vals.mean())


# --------------------------------------------------------------------------
# 3) Path fusion: waypoint 기반 클러스터링(실루엣으로 k 결정) + 근접 클러스터 병합
# --------------------------------------------------------------------------

def _pairwise_path_distance(P):
    """P: (n, n_wp, 2). 두 경로 간 거리를, 논문(Sec III-D)이 명시한 대로 '대응되는
    waypoint끼리의 유클리드 거리 평균'으로 정의한 n x n 거리행렬을 반환."""
    diff = P[:, None, :, :] - P[None, :, :, :]
    return np.linalg.norm(diff, axis=-1).mean(axis=-1)


def _kmeans_paths(P, k, n_iter=30, seed=0):
    """경로(waypoint 시퀀스) k-means. 대입/centroid 갱신 모두 waypoint 평균 거리
    기준으로 계산한다."""
    rng = np.random.default_rng(seed)
    n = len(P)
    centers = P[rng.choice(n, size=k, replace=False)].copy()
    labels = np.full(n, -1)
    for _ in range(n_iter):
        d = np.linalg.norm(P[:, None, :, :] - centers[None, :, :, :], axis=-1).mean(axis=-1)  # (n,k)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = P[m].mean(axis=0)
    return labels, centers


def _silhouette_score_paths(P, labels):
    """논문 식(2) 그대로: 실루엣 계수의 평균 (클수록 클러스터링이 좋음).
    a(i)=같은 클러스터 내 평균거리, b(i)=가장 가까운 다른 클러스터까지 평균거리."""
    D = _pairwise_path_distance(P)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return -1.0
    n = len(P)
    sil = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = min(D[i, labels == c].mean() for c in uniq if c != labels[i])
        sil[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(sil.mean())


def adaptive_kmeans_paths(P, k_max=6, seed=0):
    """실루엣 점수가 가장 높은 k를 골라 k-means를 수행 (논문: "The number of
    clusters k is determined by optimizing the silhouette loss"). 경로가 너무
    적으면(3개 미만) 클러스터링 없이 전부 하나로 묶는다."""
    n = len(P)
    k_max = min(k_max, n - 1)
    if n < 3 or k_max < 2:
        return np.zeros(n, dtype=int), P.mean(axis=0, keepdims=True)

    best_k, best_score, best_labels, best_centers = 1, -2.0, np.zeros(n, dtype=int), P.mean(axis=0, keepdims=True)
    for k in range(2, k_max + 1):
        labels, centers = _kmeans_paths(P, k, seed=seed)
        if len(np.unique(labels)) < 2:
            continue
        score = _silhouette_score_paths(P, labels)
        if score > best_score:
            best_k, best_score, best_labels, best_centers = k, score, labels, centers
    return best_labels, best_centers


def merge_close_clusters(centers, threshold_m):
    """centroid 간 평균 waypoint 거리가 threshold_m 이내인 클러스터를 하나로
    합친다. 논문은 "false positive(원래 다른 경로인데 합침, 충돌 위험) < false
    negative(원래 같은 경로인데 안 합침, 그냥 중복)"이므로 precision을 우선한다고
    명시한다 — 즉 threshold를 너무 크게 잡지 않는 쪽(잘 안 합치는 쪽)이 안전하다."""
    k = len(centers)
    parent = list(range(k))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(k):
        for j in range(i + 1, k):
            d = np.linalg.norm(centers[i] - centers[j], axis=-1).mean()
            if d <= threshold_m:
                union(i, j)

    groups = {}
    for i in range(k):
        groups.setdefault(find(i), []).append(i)
    return [centers[idxs].mean(axis=0) for idxs in groups.values()]


def path_heading_deg(path_xy):
    """경로의 대략적 진행 방향(끝점 기준 각도, +면 오른쪽)[deg]."""
    x_end, z_end = path_xy[-1]
    return float(np.degrees(np.arctan2(x_end, max(z_end, 1e-3))))


def select_final_path(merged_paths, goal_angle_deg):
    """병합된 경로가 하나면 그대로, 여러 개면 목표 방향과 heading이 가장 가까운
    경로를 선택한다 (논문 Algorithm 1의 arg min_l ∠(l, g), "Angular Selection"이
    Euclidean Selection보다 정확했다는 Table II 결과를 따름)."""
    if len(merged_paths) == 1:
        return merged_paths[0]
    diffs = [abs(((path_heading_deg(p) - goal_angle_deg + 180) % 360) - 180) for p in merged_paths]
    return merged_paths[int(np.argmin(diffs))]


# --------------------------------------------------------------------------
# 4) 시각화: BEV + 후보/선택 경로 + 목표 방향
# --------------------------------------------------------------------------

def render_paths_panel(bev_score, known_mask, interpolated_mask, meta, candidate_paths,
                        selected_path, goal_angle_deg, panel_h=480):
    """render_bev_panel()로 만든 표준 BEV 패널 위에 후보 경로(흰색, 얇게),
    선택된 경로(마젠타, 굵게 + waypoint 점), 목표 방향(패널 가장자리의 초록 점,
    논문 Fig.5의 "green dot at the edge of the map indicates the goal direction"과
    동일한 표현)을 겹쳐 그린다."""
    panel, geom = bevmod.render_bev_panel(bev_score, known_mask, interpolated_mask, meta,
                                           panel_h=panel_h, title="BEV + candidate/selected paths")
    ox, oy = geom["origin_px"]
    scale = geom["scale_px_per_m"]

    def to_px(xz):
        x, z = xz
        return (int(ox + x * scale), int(oy - z * scale))

    for p in candidate_paths:
        pts = [to_px(pt) for pt in p]
        for a, b in zip(pts[:-1], pts[1:]):
            # jet 컬러맵 배경(파랑~청록~노랑~빨강) 어디에 놓여도 잘 보이도록 흰색을 사용
            # (원래 논문 Fig.5 느낌대로 하늘색을 썼더니 배경의 청록/노랑 영역과 섞여 안 보였음)
            cv2.line(panel, a, b, (255, 255, 255), 1, cv2.LINE_AA)

    if selected_path is not None:
        # 빨강은 쓰지 않는다: BEV 배경이 jet 컬러맵이라 traversability가 높은(빨간)
        # 영역과 거의 같은 색이 되어(둘 다 BGR (0,~0-90,255) 근방) 선택 경로가 배경에
        # 묻혀 안 보이는 문제가 있었다. jet은 R,B가 동시에 크지 않으므로 마젠타는
        # 배경 어디서도 절대 나오지 않는 안전한 대비색이다.
        pts = [to_px(pt) for pt in selected_path]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(panel, a, b, (255, 0, 255), 3, cv2.LINE_AA)  # BGR: magenta
        for pt in pts:
            cv2.circle(panel, pt, 3, (255, 255, 255), -1)

    # 목표 방향: 원점에서 goal_angle_deg 방향으로 쏜 광선이 패널 경계와 만나는 점에 초록 점
    theta = np.radians(goal_angle_deg)
    dx, dy = np.sin(theta), -np.cos(theta)
    ts = []
    if dx != 0:
        ts += [(0 - ox) / dx, (panel_h - 1 - ox) / dx]
    if dy != 0:
        ts += [(0 - oy) / dy, (panel_h - 1 - oy) / dy]
    valid = [t for t in ts if t > 0 and 0 <= ox + dx * t <= panel_h - 1 and 0 <= oy + dy * t <= panel_h - 1]
    if valid:
        t = min(valid)
        goal_px = (int(ox + dx * t), int(oy + dy * t))
        cv2.circle(panel, goal_px, 7, (0, 255, 0), -1)
        cv2.circle(panel, goal_px, 7, (0, 0, 0), 1)

    cv2.putText(panel, "white=candidates  magenta=selected  green=goal dir", (geom["tick_margin"] + 10, panel_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    return panel


# --------------------------------------------------------------------------
# 5) 파이프라인 실행
# --------------------------------------------------------------------------

def process_frame(frame_bgr, frame_idx, cam_ts, ride_id, gps, out_dir, args, estimator):
    """프레임 한 장에 대해: traversability -> BEV -> 후보경로 샘플링/평가 -> 상위
    K개 클러스터링/병합 -> 목표방향 정렬 선택 -> 4분할 디버그 이미지 저장."""
    trav_map = estimator(frame_bgr)
    bev_score, known_mask, meta = bevmod.project_to_bev(
        trav_map, cam_height_m=args.cam_height, hfov_deg=args.hfov,
        cam_pitch_deg=args.cam_pitch_deg, bev_range_m=args.bev_range,
        bev_resolution_m=args.bev_resolution,
    )
    interpolated_mask = None
    if not args.no_bev_fill:
        bev_score, known_mask, interpolated_mask = bevmod.fill_bev_gaps(
            bev_score, known_mask, hfov_deg=args.hfov, bev_range_m=args.bev_range,
            max_gap_m=args.bev_fill_max_gap,
        )

    if args.goal_angle_deg is not None:
        goal_angle = args.goal_angle_deg
        goal_src = "manual"
    else:
        goal_angle = estimate_goal_angle_deg(gps, cam_ts, lookahead_s=args.goal_lookahead_s)
        goal_src = "gps" if goal_angle is not None else "gps 추정 실패->0deg 대체"
        if goal_angle is None:
            goal_angle = 0.0

    candidates = sample_candidate_paths(args.hfov, args.bev_range, n_paths=args.n_candidates,
                                         n_wp=args.n_waypoints, seed=frame_idx)
    scores = np.array([score_path_on_bev(p, bev_score, known_mask, args.bev_range) for p in candidates])
    top_idx = np.argsort(-scores)[: args.top_k]
    top_paths = candidates[top_idx]

    labels, centers = adaptive_kmeans_paths(top_paths, k_max=args.k_max, seed=frame_idx)
    merged = merge_close_clusters(centers, threshold_m=args.merge_threshold)
    selected = select_final_path(merged, goal_angle)

    paths_panel = render_paths_panel(bev_score, known_mask, interpolated_mask, meta,
                                      top_paths, selected, goal_angle)

    header = [
        f"ride={ride_id}  frame_idx={frame_idx}  estimator={args.estimator}",
        f"cam_height={args.cam_height:.2f}m  hfov={args.hfov:.0f}deg  bev_range={args.bev_range:.1f}m",
        f"candidates={args.n_candidates}  top_k={args.top_k}  clusters(raw->merged)={len(np.unique(labels))}->{len(merged)}  "
        f"goal={goal_angle:.1f}deg({goal_src})",
    ]
    heat_label = "Traversability prob. (SAM2)" if args.estimator == "sam2" else "Traversability score (heuristic)"
    panel = bevmod.render_debug_panel(frame_bgr, trav_map, bev_score, known_mask, interpolated_mask, meta,
                                       header, heat_label=heat_label, extra_panels=[paths_panel])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}_path.png")
    cv2.imwrite(out_path, panel)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", nargs="?", default=None)
    ap.add_argument("--frame-indices", type=int, nargs="+", default=None)
    ap.add_argument("--num-samples", type=int, default=6)
    # BEV (04번 스크립트와 동일 파라미터/기본값)
    ap.add_argument("--cam-height", type=float, default=0.25)
    ap.add_argument("--hfov", type=float, default=110.0)
    ap.add_argument("--cam-pitch-deg", type=float, default=0.0)
    ap.add_argument("--bev-range", type=float, default=8.0)
    ap.add_argument("--bev-resolution", type=float, default=0.05)
    ap.add_argument("--horizon-ratio", type=float, default=0.45)
    ap.add_argument("--bev-fill-max-gap", type=float, default=1.5)
    ap.add_argument("--no-bev-fill", action="store_true")
    ap.add_argument("--estimator", choices=["sam2", "heuristic"], default="sam2")
    ap.add_argument("--sam2-model", default="facebook/sam2.1-hiera-tiny")
    # 경로 계획
    ap.add_argument("--n-candidates", type=int, default=25, help="샘플링할 후보 경로 수 M")
    ap.add_argument("--n-waypoints", type=int, default=10, help="경로 1개당 waypoint 수")
    ap.add_argument("--top-k", type=int, default=8, help="BEV 비용 기준 상위 K개만 클러스터링에 사용")
    ap.add_argument("--k-max", type=int, default=6, help="adaptive k-means가 시도할 최대 클러스터 수")
    ap.add_argument("--merge-threshold", type=float, default=0.6,
                     help="클러스터 centroid 병합 임계값[m] (평균 waypoint 거리 기준)")
    ap.add_argument("--goal-lookahead-s", type=float, default=8.0,
                     help="GPS 기반 목표 방향 추정 시 몇 초 뒤 위치를 목표로 볼지")
    ap.add_argument("--goal-angle-deg", type=float, default=None,
                     help="목표 방향을 GPS 추정 대신 직접 고정값[deg]으로 지정 (+면 오른쪽)")
    ap.add_argument("--out", default=DEBUG_DIR)
    args = ap.parse_args()

    if args.session_dir is None:
        candidates = bevmod.find_ride_dirs()
        if not candidates:
            sys.exit("세션 경로를 인자로 넘겨줘: python 06_path_planning.py <ride_dir>")
        session_dir = candidates[0]
    else:
        session_dir = os.path.expanduser(args.session_dir)

    rid = bevmod.parse_ride_id(session_dir)
    sources, src_desc = bevmod.find_front_camera_sources(session_dir, rid)
    frame_ids, cam_ts_all = load_camera_timestamps(session_dir, rid)
    gps = load_gps(session_dir, rid)
    print(f"[session] {session_dir}")
    print(f"[ride_id] {rid}")
    print(f"[front_camera] {src_desc} ({len(sources)}개 소스 파일)")
    print(f"[gps] {'로딩됨 (' + str(len(gps[0])) + '개 fix)' if gps is not None else '없음 -> goal은 0deg(직진) 고정'}")

    if args.frame_indices:
        indices = sorted(set(args.frame_indices))
    else:
        total = len(frame_ids) if len(frame_ids) else 2000
        n = min(args.num_samples, total)
        indices = sorted(set(np.linspace(0, total - 1, n, dtype=int).tolist()))
    print(f"[frames] 처리할 프레임 인덱스: {indices}")

    frames = bevmod.extract_frames_at_indices(sources, indices)
    if not frames:
        sys.exit("[err] 요청한 인덱스에서 프레임을 하나도 못 읽었음")

    try:
        est_args = argparse.Namespace(estimator=args.estimator, sam2_model=args.sam2_model,
                                       horizon_ratio=args.horizon_ratio)
        estimator = bevmod.build_traversability_estimator(est_args)
    except ImportError as e:
        sys.exit(f"[err] {e}")

    out_dir = os.path.join(args.out, f"ride_{rid}")
    saved = []
    for idx in indices:
        if idx not in frames:
            print(f"[skip] frame_idx={idx} 디코딩 실패")
            continue
        cam_ts = cam_ts_all[idx] if idx < len(cam_ts_all) else None
        path = process_frame(frames[idx], idx, cam_ts, rid, gps, out_dir, args, estimator)
        saved.append(path)
        print(f"[save] {path}")

    print(f"\n[done] {len(saved)}장 저장 완료 -> {out_dir}")


if __name__ == "__main__":
    main()
