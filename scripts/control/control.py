"""
control.py
scripts/06_path_planning.py 가 매 프레임 고르는 "선택된 경로(selected path)"를 따라
실제로(또는 시뮬레이터로) 로버를 주행시키는 폐루프(closed-loop) 컨트롤러.

scripts/perception/live_bridge.py 와 구조가 같다 (SDK GET /v2/front -> traversability
-> BEV -> 후보 경로 샘플링/클러스터링/선택, 04/06번 함수 그대로 재사용) — 다만
live_bridge.py는 결과를 이미지로 렌더링해서 대시보드에 보여주는 용도였고, 이 스크립트는
그 "선택된 경로(x=우측+, z=전방+ 좌표 배열)"를 직접 조향각으로 변환해서 SDK의
POST /control 로 실제 구동 명령을 보낸다.

조향/구동 매핑 (요청대로 고정):
  - steering: 처음엔 06_path_planning.py의 path_heading_deg()(경로 "끝점" 기준 각도)를
    그대로 썼는데, 경로가 로봇 바로 앞에서는 거의 직진으로 시작해서 끝에 가서야 크게
    꺾이는 모양이라 끝점 각도를 쓰면 조향이 실제 눈앞의 굽음 정도보다 과장됐다. 그래서
    원점에서 --lookahead-m(기본 1.5m) 떨어진 지점(waypoint 사이 선형보간)까지의 각도를
    조향각으로 쓴다 (steering_deg_from_path 참고) — "살짝 굽은 경로 = 살짝만 조향"이
    되도록. -45~45도 범위로 클립한 뒤 SDK의 angular 명령값(-1~1)으로 선형 매핑한다:
    angular = -steering_deg / 45.0 (부호는 실측 기준 — 아래 참고)
  - speed: 매 사이클 고정 상수(--speed, 기본 0.5)를 그대로 linear 명령값으로 보낸다.
    원래 요청은 1.5였지만, 추론 1회에 ~350~650ms(카메라 프레임을 받아 traversability+
    경로계획까지 도는 시간)가 걸려서 그만큼 조향이 "늦게" 갱신된다 — 그 사이 로버가
    너무 멀리 가버리면 오차가 커지므로, 지연 시간을 감안해 기본 속도를 낮췄다
    (SDK 문서상 정상 범위 -1~1 안쪽이기도 함). 필요하면 --speed 로 다시 올릴 수 있다.

  *** 주의 ***: SDK README(POST /control)는 linear/angular 둘 다 공식적으로 -1~1
  범위라고 명시한다. --speed 를 1.0 초과로 주면 SDK가 서버 단에서 값을 검증/클리핑하지
  않아(main.py의 control()이 값을 그대로 RTM으로 전달) 에러 없이 전송은 되지만, 실제
  로버 펌웨어가 그 값을 어떻게 처리할지(그대로 반영/내부 클리핑/무시)는 보장이 없다.

안전장치:
  - Ctrl+C 또는 예외 발생 시 반드시 정지 명령(linear=0, angular=0)을 한 번 보내고 종료한다
    (로버가 마지막 명령을 계속 유지하며 폭주하는 것을 막기 위함).
  - 카메라 프레임을 못 받거나 추론이 실패하면, 그 사이클은 주행하지 않고(정지 명령) 다음
    사이클을 재시도한다 — perception이 불확실할 때 관성 주행하지 않기 위함.

라이브에서 06번과 다른 점(06/live_bridge와 동일한 한계, 다시 적어둠):
  - goal 방향은 06번처럼 미래 GPS로 추정할 수 없어 --goal-deg 고정값(기본 0=직진)을 쓴다.

사용:
  python3 scripts/control/control.py                       # 기본: sam_tp, speed=0.5
  python3 scripts/control/control.py --dry-run              # 명령을 SDK로 보내지 않고 출력만
  python3 scripts/control/control.py --speed 1.0 --estimator heuristic   # 더 빠르게
"""
import argparse
import base64
import importlib.util
import math
import os
import sys
import time

import cv2
import numpy as np
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_SAM_TP_CHECKPOINT = os.path.join(SCRIPTS_DIR, "perception", "models", "best_sam_tp.pt")


def _load_module(filename, name):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bevmod = _load_module("04_bev_traversability.py", "bev_traversability")
pathmod = _load_module("06_path_planning.py", "path_planning")


def fetch_front_frame(sdk_base_url, timeout=5.0):
    res = requests.get(f"{sdk_base_url}/v2/front", timeout=timeout)
    res.raise_for_status()
    b64 = res.json().get("front_frame")
    if not b64:
        return None
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def send_control(sdk_base_url, linear, angular, lamp=0, timeout=5.0):
    body = {"command": {"linear": linear, "angular": angular, "lamp": lamp}}
    res = requests.post(f"{sdk_base_url}/control", json=body, timeout=timeout)
    res.raise_for_status()
    return res.json()


# --------------------------------------------------------------------------
# 선택된 경로 -> 조향각
# --------------------------------------------------------------------------
def steering_deg_from_path(path_xy, lookahead_m, max_steer_deg):
    """path_xy: (n_wp, 2) [x=우측+, z=전방+][m], 로봇 원점(0,0) 기준.

    처음엔 06_path_planning.py의 path_heading_deg()(경로 "끝점" 기준 각도)를 그대로
    썼는데, 경로 모양이 x(t) = end_x * t^1.6~1.8 이라 로봇 바로 앞에서는 거의 직진으로
    시작해서 끝에 가서야 크게 꺾인다 — 그래서 끝점 각도를 쓰면 "지금 당장 경로가 얼마나
    굽어있는지"보다 훨씬 과장된 조향각이 나온다(살짝만 굽은 경로도 끝점 기준으로는 30~
    40도가 나올 수 있음). 그래서 대신 원점에서 --lookahead-m 만큼 떨어진 지점(waypoint
    사이 선형보간)까지의 각도를 쓴다 — 로봇 바로 앞의 경로 기울기에 비례해서 조향하므로
    "살짝 굽은 경로 = 살짝만 조향"이 된다. 경로가 lookahead보다 짧으면 마지막 waypoint를
    목표점으로 쓴다. 반환값은 -max_steer_deg~+max_steer_deg 로 클립 (+면 오른쪽)."""
    dists = np.linalg.norm(path_xy, axis=1)
    target = path_xy[-1]
    for i in range(len(path_xy) - 1):
        d0, d1 = dists[i], dists[i + 1]
        if d0 <= lookahead_m <= d1 and d1 > d0:
            frac = (lookahead_m - d0) / (d1 - d0)
            target = path_xy[i] + frac * (path_xy[i + 1] - path_xy[i])
            break
    x, z = target
    deg = math.degrees(math.atan2(x, max(z, 1e-3)))
    return float(np.clip(deg, -max_steer_deg, max_steer_deg))


def compute_path(frame, estimator, args, seed):
    """live_bridge.infer_loop 와 동일한 traversability -> BEV -> 후보경로/클러스터링/선택
    파이프라인. 렌더링은 안 하고 선택된 경로(numpy array)만 반환한다."""
    trav_map = estimator(frame)
    bev_score, known_mask, meta = bevmod.project_to_bev(
        trav_map, cam_height_m=args.cam_height, hfov_deg=args.hfov,
        cam_pitch_deg=args.cam_pitch_deg, bev_range_m=args.bev_range,
        bev_resolution_m=args.bev_resolution,
    )
    bev_score, known_mask, _interp = bevmod.fill_bev_gaps(
        bev_score, known_mask, hfov_deg=args.hfov, bev_range_m=args.bev_range,
        max_gap_m=args.bev_fill_max_gap,
    )
    candidates = pathmod.sample_candidate_paths(
        args.hfov, args.bev_range, n_paths=args.n_candidates, n_wp=args.n_waypoints, seed=seed,
    )
    scores = np.array([
        pathmod.score_path_on_bev(p, bev_score, known_mask, args.bev_range) for p in candidates
    ])
    top_idx = np.argsort(-scores)[: args.top_k]
    top_paths = candidates[top_idx]
    _labels, centers = pathmod.adaptive_kmeans_paths(top_paths, k_max=args.k_max, seed=seed)
    merged = pathmod.merge_close_clusters(centers, threshold_m=args.merge_threshold)
    return pathmod.select_final_path(merged, args.goal_deg)


def build_estimator(args):
    est_args = argparse.Namespace(
        estimator=args.estimator, sam2_model=args.sam2_model,
        horizon_ratio=args.horizon_ratio, sam_tp_checkpoint=args.sam_tp_checkpoint,
    )
    return bevmod.build_traversability_estimator(est_args)


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
                     help="목표 방향[deg] 고정값 (라이브에는 미래 GPS가 없어 06번처럼 추정 불가)")
    # 조향/구동 매핑 (요청대로)
    ap.add_argument("--max-steer-deg", type=float, default=45.0,
                     help="angular=-1~1 이 대응하는 최대 조향각[deg] (요청: -1~1 <-> -45~45deg)")
    ap.add_argument("--lookahead-m", type=float, default=1.5,
                     help="조향 목표점까지의 거리[m] (경로 끝점 대신 이만큼 앞의 기울기를 씀)")
    ap.add_argument("--speed", type=float, default=0.5,
                     help="고정 linear 명령값 (기본 0.5 — 카메라/추론 지연 감안해 낮춘 값. "
                          "SDK 문서상 정상 범위는 -1~1이니 그 이상은 주의)")
    ap.add_argument("--min-interval", type=float, default=0.0, help="제어 루프 최소 주기[초]")
    ap.add_argument("--dry-run", action="store_true", help="/control 로 실제 전송하지 않고 계산값만 출력")
    return ap


def main():
    args = build_arg_parser().parse_args()

    if abs(args.speed) > 1.0:
        print(f"[control] 경고: --speed {args.speed} 는 SDK 문서상 정상 범위(-1~1)를 벗어납니다. "
              f"그대로 전송합니다 (요청대로). 안전하게 쓰려면 --speed 1.0 으로 실행하세요.")

    print(f"[control] estimator={args.estimator}  sdk={args.sdk_base_url}  "
          f"speed(linear)={args.speed}(고정)  lookahead={args.lookahead_m}m  "
          f"max_steer={args.max_steer_deg}deg  dry_run={args.dry_run}")
    estimator = build_estimator(args)

    frame_counter = 0
    consecutive_errors = 0
    try:
        while True:
            t0 = time.time()
            try:
                frame = fetch_front_frame(args.sdk_base_url)
                if frame is None:
                    raise RuntimeError("front_frame 없음 (SDK가 아직 카메라 프레임을 못 받은 상태)")

                selected = compute_path(frame, estimator, args, seed=frame_counter)
                steer_deg = steering_deg_from_path(selected, args.lookahead_m, args.max_steer_deg)
                # SDK 예제(examples/basics/02_diagonal_movement.py)는 angular>0=오른쪽이라고
                # 하지만, 실제 로버에서 반대로 도는 게 확인돼서 부호를 뒤집는다 (실측 우선).
                angular = -float(np.clip(steer_deg / args.max_steer_deg, -1.0, 1.0))
                linear = args.speed

                infer_ms = (time.time() - t0) * 1000.0
                print(f"[control] frame={frame_counter}  infer={infer_ms:.0f}ms  "
                      f"steer={steer_deg:+.1f}deg  angular={angular:+.3f}  linear={linear:+.2f}")

                if not args.dry_run:
                    send_control(args.sdk_base_url, linear, angular)

                frame_counter += 1
                consecutive_errors = 0

            except Exception as e:  # noqa: BLE001 - perception/전송 실패 시 정지하고 재시도
                consecutive_errors += 1
                print(f"[control] 에러: {e} -> 정지 명령 후 재시도")
                if not args.dry_run:
                    try:
                        send_control(args.sdk_base_url, 0.0, 0.0)
                    except Exception as stop_err:  # noqa: BLE001
                        print(f"[control] 정지 명령도 실패: {stop_err}")
                time.sleep(min(5.0, 0.5 * consecutive_errors))
                continue

            elapsed = time.time() - t0
            if args.min_interval > elapsed:
                time.sleep(args.min_interval - elapsed)

    except KeyboardInterrupt:
        print("\n[control] 종료 요청 -> 정지 명령 전송")
        if not args.dry_run:
            try:
                send_control(args.sdk_base_url, 0.0, 0.0)
            except Exception as e:  # noqa: BLE001
                print(f"[control] 정지 명령 실패: {e}")
        sys.exit(0)


if __name__ == "__main__":
    main()
