"""
live_bridge.py
라이브 Earth Rover 전면 카메라 프레임에 대해 traversability + BEV + path planning을
계속 돌리는 추론 엔진. HTTP 서버는 갖고 있지 않다 — scripts/sensors/serve_dashboard.py가
이 모듈을 import 해서 백그라운드 스레드로 돌리고, 자기 자신의 (대시보드와 같은) 포트에서
결과를 서빙한다. (전에는 이 파일이 자체 HTTP 서버로 9222 포트에 따로 떴었는데, 대시보드
포트 하나로 합쳐달라는 요청에 따라 이제는 순수 엔진 모듈이다.)

scripts/04_bev_traversability.py / scripts/06_path_planning.py 는 "녹화된 세션 파일"을
대상으로 하는 배치 CLI라서 라이브 프레임을 받는 코드가 없다. 여기서는 그 두 스크립트의
함수(estimator, BEV projection, path sampling/fusion, BEV 렌더링)를 그대로 재사용해서:
  1) SDK의 GET /v2/front를 계속 당겨온다.
  2) traversability -> BEV projection -> 후보 경로 샘플링/클러스터링/선택까지 06번과
     동일한 파이프라인을 돌린다.
  3) 결과를 이미지 2장으로 만든다:
       - front_image: 원본 전면 카메라 위에 traversability 히트맵을 반투명으로 얹고,
         선택된/후보 경로를 원근 투영해서 실제 화면 위에 그린 "AR 오버레이" 이미지.
         (04/06번은 이 투영을 안 하고 옆에 별도 BEV 패널로만 보여줬었다. project_to_bev()가
         쓰는 핀홀+지면 모델의 역변환(project_ground_to_pixel)을 새로 추가해서, 선택된
         경로의 (x=우측, z=전방)[m] 좌표를 다시 카메라 픽셀로 되돌려 그린다.)
       - bev_image: 06번과 동일한 새눈(bird's-eye) 시점 BEV cost map + 후보/선택 경로.
  4) 둘 다 latest(LatestResult)에 JPEG base64로 저장해두고, HTTP 쪽(serve_dashboard.py)은
     요청이 올 때마다 이 최신 결과를 즉시 반환한다 (요청마다 추론 X, 폴링 속도와 추론
     속도를 분리 — 대시보드가 자주 물어봐도 추론 큐가 안 쌓임).

라이브에서 06번과 다른 점 (읽어볼 것):
  - 06번의 목표 방향(goal angle)은 "몇 초 뒤 실제 GPS가 어디로 갔는지"를 사후에 보고
    계산하는 방식이라 라이브에는 쓸 수 없다(미래가 없음). 여기서는 --goal-deg 로 고정값
    (기본 0=직진)을 준다.
  - 매 프레임 추론 비용이 있으므로(특히 sam_tp/sam2), 대시보드의 다른 폴링(0.15~0.5초)
    만큼 빠르지 않을 수 있다. infer_ms를 응답에 같이 실어서 실제 속도를 그대로 보여준다.
"""
import argparse
import base64
import importlib.util
import math
import os
import threading
import time

import cv2
import numpy as np
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_SAM_TP_CHECKPOINT = os.path.join(SCRIPT_DIR, "models", "best_sam_tp.pt")


def _load_module(filename, name):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bevmod = _load_module("04_bev_traversability.py", "bev_traversability")
pathmod = _load_module("06_path_planning.py", "path_planning")


# --------------------------------------------------------------------------
# 공유 상태: 백그라운드 추론 루프가 계속 갱신하고, HTTP 핸들러는 읽기만 한다.
# --------------------------------------------------------------------------
class LatestResult:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {"status": "starting"}

    def set(self, data):
        with self._lock:
            self._data = data

    def get(self):
        with self._lock:
            return dict(self._data)


latest = LatestResult()


def fetch_front_frame(sdk_base_url, timeout=5.0):
    """SDK GET /v2/front 에서 최신 전면 프레임을 받아 BGR numpy 이미지로 디코딩."""
    res = requests.get(f"{sdk_base_url}/v2/front", timeout=timeout)
    res.raise_for_status()
    b64 = res.json().get("front_frame")
    if not b64:
        return None
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_jpeg_b64(image_bgr, quality=80):
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 인코딩 실패")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# --------------------------------------------------------------------------
# project_to_bev()의 역변환: BEV 지면좌표(x=우측+, z=전방+)[m] -> 카메라 픽셀(u,v)
# --------------------------------------------------------------------------
def project_ground_to_pixel(x_lateral_m, z_forward_m, cam_height_m, hfov_deg, cam_pitch_deg, img_w, img_h):
    """project_to_bev()가 픽셀 -> 지면점을 구하던 것과 정확히 반대 방향 계산.
    반환: (u,v) 픽셀 좌표, 또는 카메라 뒤쪽/화면 밖이면 None.

    원리: 지면점 (x_lateral, cam_height, z_forward)는 카메라 원점에서 나가는 광선 위에
    있으므로, 이 벡터를 정규화하면 그 광선의 (world-aligned, 즉 pitch 보정된) 방향과
    같다. project_to_bev()가 했던 pitch 회전의 역회전을 적용해 원래 카메라 좌표계의
    방향으로 되돌린 뒤, 표준 핀홀 투영(u=cx+fx*x/z, v=cy+fy*y/z)으로 픽셀을 얻는다."""
    hfov = math.radians(hfov_deg)
    fx = (img_w / 2.0) / math.tan(hfov / 2.0)
    fy = fx
    cx, cy = img_w / 2.0, img_h / 2.0

    v = np.array([x_lateral_m, cam_height_m, z_forward_m], dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    dx, dy_r, dz_r = v / n

    pitch = math.radians(cam_pitch_deg)
    dy = dy_r * math.cos(pitch) - dz_r * math.sin(pitch)
    dz = dy_r * math.sin(pitch) + dz_r * math.cos(pitch)

    if dz <= 1e-4:  # 카메라 뒤쪽(또는 지평선) -> 투영 불가
        return None

    u = cx + fx * (dx / dz)
    pv = cy + fy * (dy / dz)
    return (u, pv)


def _draw_path_on_image(img, path_xy, cam_height_m, hfov_deg, cam_pitch_deg, color, thickness, dot_radius=0):
    h, w = img.shape[:2]
    pts = []
    for x, z in path_xy:
        px = project_ground_to_pixel(x, z, cam_height_m, hfov_deg, cam_pitch_deg, w, h)
        pts.append(px)
    for a, b in zip(pts[:-1], pts[1:]):
        if a is None or b is None:
            continue
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, thickness, cv2.LINE_AA)
    if dot_radius:
        for p in pts:
            if p is not None:
                cv2.circle(img, (int(p[0]), int(p[1])), dot_radius, (255, 255, 255), -1)


def render_front_overlay(frame_bgr, trav_map, candidate_paths, selected_path,
                          cam_height_m, hfov_deg, cam_pitch_deg, alpha=0.35):
    """원본 전면 카메라 위에 traversability 히트맵(반투명) + 후보경로(흰색) + 선택경로
    (마젠타)를 실제 원근으로 투영해서 그린 "AR 오버레이" 이미지."""
    heat = bevmod._colorize(trav_map)
    overlay = cv2.addWeighted(heat, alpha, frame_bgr, 1.0 - alpha, 0)

    for p in candidate_paths:
        _draw_path_on_image(overlay, p, cam_height_m, hfov_deg, cam_pitch_deg, (255, 255, 255), 1)
    if selected_path is not None:
        _draw_path_on_image(overlay, selected_path, cam_height_m, hfov_deg, cam_pitch_deg,
                             (255, 0, 255), 3, dot_radius=3)

    cv2.putText(overlay, "Front + traversability + path", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


# --------------------------------------------------------------------------
# 백그라운드 추론 루프
# --------------------------------------------------------------------------
def infer_loop(args, estimator):
    """계속 프레임을 받아 traversability -> BEV -> path planning -> 오버레이 2장 렌더링을
    반복하고, 매번 latest 를 갱신한다. 프레임 하나 처리에 걸리는 시간이 곧 이 루프의
    주기이므로 별도 sleep 없이 바로 다음 프레임으로 넘어간다 (--min-interval 로 최소
    주기를 둘 수 있음 — 가벼운 추정기가 SDK/CPU를 과도하게 두드리지 않게)."""
    frame_counter = 0
    consecutive_errors = 0

    while True:
        t0 = time.time()
        try:
            frame = fetch_front_frame(args.sdk_base_url)
            if frame is None:
                raise RuntimeError("front_frame 없음 (SDK가 아직 카메라 프레임을 못 받은 상태)")

            trav_map = estimator(frame)
            bev_score, known_mask, meta = bevmod.project_to_bev(
                trav_map, cam_height_m=args.cam_height, hfov_deg=args.hfov,
                cam_pitch_deg=args.cam_pitch_deg, bev_range_m=args.bev_range,
                bev_resolution_m=args.bev_resolution,
            )
            bev_score, known_mask, interpolated_mask = bevmod.fill_bev_gaps(
                bev_score, known_mask, hfov_deg=args.hfov, bev_range_m=args.bev_range,
                max_gap_m=args.bev_fill_max_gap,
            )

            candidates = pathmod.sample_candidate_paths(
                args.hfov, args.bev_range, n_paths=args.n_candidates,
                n_wp=args.n_waypoints, seed=frame_counter,
            )
            scores = np.array([
                pathmod.score_path_on_bev(p, bev_score, known_mask, args.bev_range)
                for p in candidates
            ])
            top_idx = np.argsort(-scores)[: args.top_k]
            top_paths = candidates[top_idx]
            _labels, centers = pathmod.adaptive_kmeans_paths(top_paths, k_max=args.k_max, seed=frame_counter)
            merged = pathmod.merge_close_clusters(centers, threshold_m=args.merge_threshold)
            selected = pathmod.select_final_path(merged, args.goal_deg)

            bev_image = pathmod.render_paths_panel(
                bev_score, known_mask, interpolated_mask, meta, top_paths, selected, args.goal_deg,
            )
            front_image = render_front_overlay(
                frame, trav_map, top_paths, selected,
                args.cam_height, args.hfov, args.cam_pitch_deg,
            )

            infer_ms = (time.time() - t0) * 1000.0
            latest.set({
                "status": "ok",
                "front_image": encode_jpeg_b64(front_image, quality=args.jpeg_quality),
                "bev_image": encode_jpeg_b64(bev_image, quality=args.jpeg_quality),
                "estimator": args.estimator,
                "infer_ms": round(infer_ms, 1),
                "frame": frame_counter,
                "goal_deg": args.goal_deg,
            })
            frame_counter += 1
            consecutive_errors = 0

        except Exception as e:  # noqa: BLE001 - 브릿지는 죽지 않고 계속 재시도해야 함
            consecutive_errors += 1
            latest.set({"status": "error", "error": str(e)})
            time.sleep(min(5.0, 0.5 * consecutive_errors))
            continue

        elapsed = time.time() - t0
        if args.min_interval > elapsed:
            time.sleep(args.min_interval - elapsed)


# --------------------------------------------------------------------------
# CLI 인자 정의 (serve_dashboard.py가 parents=[...]로 재사용)
# --------------------------------------------------------------------------
def build_arg_parser():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--sdk-base-url", default="http://localhost:8000")
    ap.add_argument("--estimator", choices=["heuristic", "sam2", "sam_tp"], default="sam_tp")
    ap.add_argument("--sam2-model", default="facebook/sam2.1-hiera-tiny")
    ap.add_argument("--sam-tp-checkpoint", default=DEFAULT_SAM_TP_CHECKPOINT)
    ap.add_argument("--horizon-ratio", type=float, default=0.45)
    ap.add_argument("--cam-height", type=float, default=0.25)
    ap.add_argument("--hfov", type=float, default=110.0)
    ap.add_argument("--cam-pitch-deg", type=float, default=0.0)
    ap.add_argument("--bev-range", type=float, default=8.0)
    ap.add_argument("--bev-resolution", type=float, default=0.05)
    ap.add_argument("--bev-fill-max-gap", type=float, default=1.5)
    ap.add_argument("--n-candidates", type=int, default=25)
    ap.add_argument("--n-waypoints", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--k-max", type=int, default=6)
    ap.add_argument("--merge-threshold", type=float, default=0.6)
    ap.add_argument("--goal-deg", type=float, default=0.0,
                     help="목표 방향[deg] 고정값 (라이브에는 미래 GPS가 없어 06번처럼 추정 불가, +면 오른쪽)")
    ap.add_argument("--min-interval", type=float, default=0.0,
                     help="추론 루프 최소 주기[초] (가벼운 추정기가 CPU를 계속 100%% 잡아먹지 않게)")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    return ap


def build_estimator(args):
    est_args = argparse.Namespace(
        estimator=args.estimator, sam2_model=args.sam2_model,
        horizon_ratio=args.horizon_ratio, sam_tp_checkpoint=args.sam_tp_checkpoint,
    )
    return bevmod.build_traversability_estimator(est_args)
