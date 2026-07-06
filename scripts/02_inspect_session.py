"""
02_inspect_session.py
FrodoBots-2K 세션 하나를 로드해서 (1) 센서 확인 (2) 카메라 프레임 추출
(3) RPM->속도 변환 (4) 관측->제어 페어(behavior cloning 재료) (5) GPS 궤적 시각화.

컬럼/파일 구조는 공식 repo(catglossop/frodo_dataset)의 helpercode.ipynb 및
convert_frodo_to_gnm_vGPS_and_rpm.py 기준으로 확정한 값이라 추측이 아님.

사용:
  pip install pandas numpy matplotlib opencv-python
  python 02_inspect_session.py "../data/frodobots-dataset-getting-started/ride_19154_20240225023555"
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data", "frodobots-dataset-getting-started")

# 로봇 상수 (공식 변환 스크립트 값)
ROBOT_WHEEL_R = 0.065   # 바퀴 반지름 [m]
ROBOT_L = 0.206         # 좌우 트랙 폭 [m]


def rpm_to_vw(rpm1, rpm2, rpm3, rpm4):
    """4바퀴 RPM -> 선속도 v[m/s], 각속도 w[rad/s].  좌=1,3 / 우=2,4"""
    v_l = (rpm1 + rpm3) * np.pi * ROBOT_WHEEL_R / 60.0
    v_r = (rpm2 + rpm4) * np.pi * ROBOT_WHEEL_R / 60.0
    v = (v_r + v_l) / 2.0
    w = (v_r - v_l) / ROBOT_L
    return v, w


def load_session(session_dir):
    rid = os.path.basename(session_dir.rstrip("/")).split("_")[1]
    p = lambda name: os.path.join(session_dir, f"{name}_{rid}.csv")

    control = pd.read_csv(p("control_data"))
    gps = pd.read_csv(p("gps_data"))
    imu = pd.read_csv(p("imu_data"))
    front_ts = pd.read_csv(p("front_camera_timestamps"))
    front_mp4 = os.path.join(session_dir, f"front_camera_{rid}.mp4")
    return rid, control, gps, imu, front_ts, front_mp4


def unpack_imu(imu):
    """imu의 compass/gyroscope/accelerometer 셀(JSON 리스트)을 평평한 df로."""
    out = {}
    for key in ["accelerometer", "gyroscope", "compass"]:
        rows = []
        if key in imu.columns:
            for cell in imu[key].dropna():
                try:
                    rows.extend(json.loads(cell))
                except (json.JSONDecodeError, TypeError):
                    pass
        out[key] = pd.DataFrame(rows)
    return out


def build_obs_action_pairs(control, front_ts):
    """카메라 프레임 timestamp마다 가장 가까운 control 행을 매칭 -> (frame_idx, action).
    action = (linear, angular) 게임 입력 + (v_rpm, w_rpm) 실측 속도."""
    # timestamp 컬럼 자동 탐지 (앞의 head 출력으로 단위 꼭 확인)
    ts_col_ctrl = "timestamp"
    ts_col_cam = "timestamp" if "timestamp" in front_ts.columns else front_ts.columns[0]

    ctrl_ts = control[ts_col_ctrl].to_numpy()
    v_rpm, w_rpm = rpm_to_vw(control.rpm_1, control.rpm_2, control.rpm_3, control.rpm_4)

    pairs = []
    for fidx, t in enumerate(front_ts[ts_col_cam].to_numpy()):
        j = int(np.argmin(np.abs(ctrl_ts - t)))
        pairs.append({
            "frame_idx": fidx,
            "cam_ts": t,
            "ctrl_ts": ctrl_ts[j],
            "linear": float(control.linear.iloc[j]),    # 게임 입력(정답 라벨 후보)
            "angular": float(control.angular.iloc[j]),
            "v_rpm": float(v_rpm.iloc[j]),               # RPM에서 나온 실제 속도
            "w_rpm": float(w_rpm.iloc[j]),
        })
    return pd.DataFrame(pairs)


def main(session_dir):
    session_dir = os.path.expanduser(session_dir)
    print(f"[session] {session_dir}")
    rid, control, gps, imu, front_ts, front_mp4 = load_session(session_dir)
    print(f"[ride_id] {rid}\n")

    # 1) 센서 미리보기 (timestamp 단위 여기서 눈으로 확인!)
    print("=== control_data.head() ===\n", control.head(), "\n")
    print("control 컬럼:", list(control.columns))
    print("=== gps_data.head() ===\n", gps.head(), "\n")
    imu_dfs = unpack_imu(imu)
    for k, d in imu_dfs.items():
        print(f"[imu] {k}: {len(d)} rows, 컬럼 {list(d.columns)}")

    # 2) 전면 카메라 프레임 하나 뽑아 저장
    frame_id = 5
    cap = cv2.VideoCapture(front_mp4)
    frame = None
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == frame_id:
            frame = f
            break
    cap.release()
    if frame is not None:
        cv2.imwrite("sample_front_frame.jpg", frame)
        print(f"\n[camera] 프레임 {frame_id} -> sample_front_frame.jpg "
              f"({frame.shape[1]}x{frame.shape[0]})")

    # 3) & 4) 관측->제어 페어 (behavior cloning 재료)
    pairs = build_obs_action_pairs(control, front_ts)
    pairs.to_csv("obs_action_pairs.csv", index=False)
    print(f"\n[pairs] 프레임-제어 페어 {len(pairs)}개 -> obs_action_pairs.csv")
    print(pairs.head())

    # 5) GPS 궤적 + 속도 프로파일 그림
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(gps.longitude, gps.latitude, ".-", ms=2)
    ax[0].set_title(f"GPS trajectory (ride {rid})")
    ax[0].set_xlabel("longitude"); ax[0].set_ylabel("latitude")
    ax[0].axis("equal")

    v_rpm, w_rpm = rpm_to_vw(control.rpm_1, control.rpm_2, control.rpm_3, control.rpm_4)
    ax[1].plot(control.timestamp, v_rpm, label="v (m/s, from RPM)")
    ax[1].plot(control.timestamp, w_rpm, label="w (rad/s, from RPM)")
    ax[1].plot(control.timestamp, control.linear, "--", alpha=.6, label="linear (input)")
    ax[1].plot(control.timestamp, control.angular, "--", alpha=.6, label="angular (input)")
    ax[1].set_title("Control / velocity"); ax[1].set_xlabel("timestamp"); ax[1].legend()

    fig.tight_layout()
    fig.savefig("session_overview.png", dpi=120)
    print("[plot] -> session_overview.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 인자 없으면 data/frodobots-dataset-getting-started 안에서 자동 탐색
        cand = sorted(glob.glob(os.path.join(DATA_DIR, "**", "ride_*"), recursive=True))
        cand = [c for c in cand if os.path.isdir(c)]
        if not cand:
            print("세션 경로를 인자로 넘겨줘: python 02_inspect_session.py <ride_dir>")
            sys.exit(1)
        main(cand[0])
    else:
        main(sys.argv[1])
