"""
05_train_bev_policy.py
04_bev_traversability.py 로 만드는 BEV traversability cost map "한 장만" 입력으로
받아서, 그 상황에서 실제 로버가 어떻게 조종됐는지(linear, angular)를 예측하도록
학습하는 간단한 behavior-cloning 실험 스크립트.

[왜 만들었나]
GeNIE 논문의 핵심 주장(Sec III-D)은 "BEV cost map 위에서 traversability가 높고
목표 방향과 정렬된 경로를 고른다"이다. 이 스크립트는 그 주장을 이 데이터셋 위에서
아주 단순하게 검증해보기 위한 것이다: 카메라 원본 이미지가 아니라 "BEV cost map
한 장"만 보고도 실제 사람이 조종한 (linear, angular)를 어느 정도 맞출 수 있는지
학습/평가한다. 잘 맞는다면 BEV cost map이 조향에 필요한 정보를 충분히 담고 있다는
방증이 되고, 학습된 모델이 정말로 traversability가 높은 쪽을 가리키는 예측을
하는지는 "정성 평가" 단계에서 BEV 위에 화살표로 정답/예측을 같이 그려 눈으로도
확인한다.

파이프라인:
  1) build_bev_dataset(): 여러 ride를 순회하며 프레임을 --frame-stride 간격으로
     샘플링 -> 04_bev_traversability.py의 estimate_traversability / project_to_bev
     를 그대로 재사용해 BEV cost map을 만들고, 그 프레임 시각에 가장 가까운
     control_data 행의 (linear, angular)를 정답 라벨로 붙인다 (매칭 로직은
     02_inspect_session.py의 build_obs_action_pairs와 동일하게 최근접 timestamp).
     결과를 .npz로 캐싱해서 재실행 시 다시 계산하지 않게 한다.
  2) BevPolicyNet: 1채널(BEV 확률/코스트 맵) 입력 -> 작은 CNN -> [linear, angular]
     회귀. 03b_train_bc.py와 동일하게 tanh로 [-1,1] bound, angular 쪽 손실에
     가중치를 줘서 작은 값이 학습에서 무시되지 않게 한다.
  3) 시간순 val 홀드아웃(ride별로 마지막 --val-frac 구간)을 써서, 인접 프레임이
     거의 같은 그림이라 랜덤분할이면 val 점수가 낙관적으로 나오는 문제를 피한다.
  4) 학습 후 검증셋 일부를 골라 BEV 위에 "정답 방향(초록)"과 "예측 방향(마젠타)"
     화살표를 같이 그려 debug/bev_policy_eval/ 에 저장한다 -> 예측이 traversability
     가 높은(빨간) 쪽을 향하는지 눈으로 확인하기 위한 정성 평가.

사용:
  # 1) 데이터셋 캐시 생성 + 학습 (heuristic 추정기, torch만 있으면 됨, 빠름)
  python3 scripts/05_train_bev_policy.py --epochs 20

  # 2) SAM2 추정기로 (더 정확하지만 훨씬 느림 -> 프레임 수를 줄여서 실행 권장)
  /home/kante/miniconda3/envs/isaaclab51/bin/python3 scripts/05_train_bev_policy.py \
      --estimator sam2 --frame-stride 60 --max-frames-per-ride 80 --epochs 20

  # 캐시를 이미 만들어놨으면 --rebuild 없이 재실행 시 바로 로딩만 하고 학습 시작
필요 패키지: numpy, opencv-python, torch, matplotlib  (+ sam2 추정기 쓰려면 transformers)
"""
import os
import sys
import glob
import csv
import time
import argparse
import importlib.util

import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
DEBUG_DIR = os.path.join(REPO_ROOT, "debug")
RUNS_DIR = os.path.join(REPO_ROOT, "runs", "bev_policy")
CACHE_DEFAULT = os.path.join(REPO_ROOT, "bev_policy_data", "dataset.npz")


def _load_bev_module():
    """04_bev_traversability.py는 파일명이 숫자로 시작해서 일반 `import` 문으로 못
    불러오므로, 파일 경로를 기준으로 importlib로 직접 로드해서 재사용한다."""
    path = os.path.join(SCRIPT_DIR, "04_bev_traversability.py")
    spec = importlib.util.spec_from_file_location("bev_traversability", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bevmod = _load_bev_module()


# --------------------------------------------------------------------------
# 1) 데이터셋 생성: (BEV cost map, [linear, angular]) 쌍 수집
# --------------------------------------------------------------------------

def find_ride_dirs_for_training(patterns):
    """--rides 인자(경로 또는 glob 패턴 리스트)를 실제 ride 폴더 리스트로 펼친다.
    인자가 없으면 04_bev_traversability.find_ride_dirs()로 data/ 아래 전부를 찾는다."""
    if not patterns:
        return bevmod.find_ride_dirs()
    out = []
    for p in patterns:
        p = os.path.expanduser(p)
        matches = sorted(d for d in glob.glob(p) if os.path.isdir(d))
        if not matches and os.path.isdir(p):
            matches = [p]
        out.extend(matches)
    seen, uniq = set(), []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def load_control(session_dir, rid):
    """control_data_<rid>.csv -> (timestamp[s], linear, angular) numpy 배열."""
    path = os.path.join(session_dir, f"control_data_{rid}.csv")
    ts, lin, ang = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts.append(float(row["timestamp"]))
            lin.append(float(row["linear"]))
            ang.append(float(row["angular"]))
    return np.asarray(ts), np.asarray(lin, dtype=np.float32), np.asarray(ang, dtype=np.float32)


def load_camera_timestamps(session_dir, rid):
    """front_camera_timestamps_<rid>.csv -> (frame_id, timestamp[s]) numpy 배열."""
    path = os.path.join(session_dir, f"front_camera_timestamps_{rid}.csv")
    frame_ids, ts = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            frame_ids.append(int(row["frame_id"]))
            ts.append(float(row["timestamp"]))
    return np.asarray(frame_ids), np.asarray(ts)


def build_bev_dataset(ride_dirs, args):
    """여러 ride를 순회하며 (BEV cost map, [linear, angular]) 쌍을 모은다.

    프레임은 --frame-stride 간격으로 건너뛰며 뽑고, ride당 --max-frames-per-ride로
    상한을 둔다 (ride 15개를 전부 촘촘히 쓰면 SAM2 기준 몇 시간이 걸릴 수 있어
    기본값은 보수적으로 잡았다). 프레임의 카메라 timestamp와 가장 가까운
    control_data 행을 찾아 그 (linear, angular)를 정답 라벨로 쓴다.
    """
    import argparse as _argparse
    est_args = _argparse.Namespace(estimator=args.estimator, sam2_model=args.sam2_model,
                                    horizon_ratio=args.horizon_ratio)
    estimator = bevmod.build_traversability_estimator(est_args)

    X_list, y_list, ride_list, idx_list = [], [], [], []
    for session_dir in ride_dirs:
        rid = bevmod.parse_ride_id(session_dir)
        try:
            sources, _ = bevmod.find_front_camera_sources(session_dir, rid)
            ctrl_ts, ctrl_lin, ctrl_ang = load_control(session_dir, rid)
            frame_ids, cam_ts = load_camera_timestamps(session_dir, rid)
        except (FileNotFoundError, OSError) as e:
            print(f"[skip ride] {session_dir}: {e}")
            continue

        n_total = len(frame_ids)
        idxs = list(range(0, n_total, args.frame_stride))[: args.max_frames_per_ride]
        print(f"[ride {rid}] 총 {n_total}프레임 중 {len(idxs)}개 샘플링 (stride={args.frame_stride})")

        t0 = time.time()
        frames = bevmod.extract_frames_at_indices(sources, idxs)
        n_ok = 0
        for i in idxs:
            if i not in frames:
                continue
            j = int(np.argmin(np.abs(ctrl_ts - cam_ts[i])))
            trav = estimator(frames[i])
            bev, known, _ = bevmod.project_to_bev(
                trav, cam_height_m=args.cam_height, hfov_deg=args.hfov,
                cam_pitch_deg=args.cam_pitch_deg, bev_range_m=args.bev_range,
                bev_resolution_m=args.bev_resolution,
            )
            if not args.no_bev_fill:
                bev, known, _ = bevmod.fill_bev_gaps(
                    bev, known, hfov_deg=args.hfov, bev_range_m=args.bev_range,
                    max_gap_m=args.bev_fill_max_gap,
                )
            X_list.append(bev.astype(np.float32))
            y_list.append([ctrl_lin[j], ctrl_ang[j]])
            ride_list.append(rid)
            idx_list.append(i)
            n_ok += 1
        print(f"[ride {rid}] {n_ok}개 처리, {time.time() - t0:.1f}초")

    if not X_list:
        sys.exit("[err] 수집된 샘플이 하나도 없음 (ride 경로/CSV를 확인해줘)")

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y, np.asarray(ride_list), np.asarray(idx_list)


def load_or_build_dataset(args):
    """--cache 경로에 이미 데이터셋이 있으면(그리고 --rebuild가 아니면) 그냥 불러오고,
    없으면 build_bev_dataset()으로 새로 만들어서 캐시에 저장한다."""
    if os.path.isfile(args.cache) and not args.rebuild:
        print(f"[cache] {args.cache} 에서 불러옴 (다시 만들려면 --rebuild)")
        d = np.load(args.cache, allow_pickle=True)
        return d["X"], d["y"], d["ride"], d["idx"]

    ride_dirs = find_ride_dirs_for_training(args.rides)
    if not ride_dirs:
        sys.exit("학습에 쓸 ride 폴더를 못 찾음. --rides 로 직접 지정해줘.")
    print(f"[dataset] ride {len(ride_dirs)}개에서 BEV 데이터셋 생성 중 (estimator={args.estimator}) ...")
    t0 = time.time()
    X, y, ride, idx = build_bev_dataset(ride_dirs, args)
    print(f"[dataset] 샘플 {len(X)}개, 총 {time.time() - t0:.1f}초 소요")

    cache_dir = os.path.dirname(args.cache)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(args.cache, X=X, y=y, ride=ride, idx=idx)
    print(f"[cache] {args.cache} 에 저장")
    return X, y, ride, idx


# --------------------------------------------------------------------------
# 2) 모델 / 학습
# --------------------------------------------------------------------------

def _lazy_import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        return torch, nn, Dataset, DataLoader
    except ImportError as e:
        sys.exit(f"[err] 학습에는 torch가 필요합니다. pip install torch 로 설치해줘. (원인: {e})")


torch, nn, Dataset, DataLoader = _lazy_import_torch()


class BevGridDataset(Dataset):
    """BEV cost map(HxW, [0,1] float32)을 1채널 이미지로 보고 [linear, angular]를
    맞추는 학습용 Dataset."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).unsqueeze(0)  # (1,H,W)
        y = torch.from_numpy(self.y[i])
        return x, y


class BevPolicyNet(nn.Module):
    """BEV cost map -> [linear, angular] 회귀 네트워크.
    04번 스크립트 기본값(8m/5cm=160x160)처럼 격자가 이미 작고 채널도 1개뿐이라,
    ResNet 같은 무거운 백본 없이 conv 3층 + FC 2층으로 충분하다."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # H/2
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # H/4
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # H/8
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 2), nn.Tanh(),  # 출력 [-1,1] (linear, angular 정규화 범위와 일치)
        )

    def forward(self, x):
        return self.head(self.features(x))


def time_ordered_split(ride_arr, val_frac):
    """03b_train_bc.py와 같은 '마지막 val_frac을 val로' 방식을 ride별로 적용해서
    합친다. 여러 ride를 이어붙인 배열을 통째로 뒤에서 자르면 뒤쪽 ride 전체가
    val이 되어버리므로, ride 각각의 시간순 안에서 나눈 뒤 합쳐야 한다."""
    train_idx, val_idx = [], []
    for rid in np.unique(ride_arr):
        idxs = np.where(ride_arr == rid)[0]  # build_bev_dataset에서 프레임 순서대로 채워짐
        cut = max(1, int(len(idxs) * (1 - val_frac)))
        train_idx.extend(idxs[:cut])
        val_idx.extend(idxs[cut:])
    return np.asarray(train_idx), np.asarray(val_idx)


def run_epoch(model, loader, w_ang, device, opt=None):
    """03b_train_bc.py의 run_epoch와 동일한 패턴: angular 손실에 w_ang 가중치를 줘서
    (angular 값이 linear보다 훨씬 작으므로) 학습에서 무시되지 않게 한다."""
    train = opt is not None
    model.train(train)
    tot, mae = 0.0, np.zeros(2)
    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        per = ((pred - y) ** 2).mean(0)  # [lin, ang] 별 MSE
        loss = per[0] + w_ang * per[1]
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        bs = x.size(0)
        tot += loss.item() * bs
        mae += (pred - y).abs().mean(0).detach().cpu().numpy() * bs
    n = len(loader.dataset)
    return tot / n, mae / n


# --------------------------------------------------------------------------
# 3) 정성 평가: BEV 위에 정답/예측 방향 화살표 그리기
# --------------------------------------------------------------------------

def action_to_arrow(action, max_look_m=3.0, max_turn_deg=35.0):
    """(linear, angular) 정규화 값(-1~1)을 BEV 평면 위 화살표 (dx, dz)[m]로 변환.

    주의: 이건 실제 물리 단위 변환식이 아니라 "이 명령이 대략 어느 방향을
    향하는지"만 눈으로 보려는 정성적(qualitative) 근사다. 로버의 실제 조향각
    캘리브레이션 값이 없으므로, angular in [-1,1] -> 회전각 [-max_turn_deg,
    +max_turn_deg]로, linear in [-1,1] -> 화살표 길이(전진일 때만, 후진/정지는
    아주 짧게 표시)로 선형 매핑한다."""
    linear, angular = float(action[0]), float(action[1])
    heading_deg = np.clip(angular, -1.0, 1.0) * max_turn_deg
    length_m = max(0.3, linear) * max_look_m if linear > 0 else 0.3
    theta = np.radians(heading_deg)
    dx = length_m * np.sin(theta)  # + 면 오른쪽
    dz = length_m * np.cos(theta)  # 전방
    return dx, dz


def render_policy_eval(bev_score, true_action, pred_action, bev_range_m, panel=480):
    """BEV cost map 위에 정답(초록)/예측(마젠타) 방향 화살표를 같이 그려서, 학습된
    정책이 traversability가 높은(빨간) 쪽을 향하는지 눈으로 확인할 수 있게 한다."""
    color = bevmod._colorize(bev_score)
    img = cv2.resize(color, (panel, panel), interpolation=cv2.INTER_NEAREST)
    scale = panel / bev_range_m
    origin = (panel // 2, panel - 4)

    def draw(action, bgr):
        dx, dz = action_to_arrow(action)
        pt = (int(origin[0] + dx * scale), int(origin[1] - dz * scale))
        pt = (int(np.clip(pt[0], 0, panel - 1)), int(np.clip(pt[1], 0, panel - 1)))
        cv2.arrowedLine(img, origin, pt, bgr, 2, tipLength=0.25)

    draw(true_action, (0, 255, 0))
    draw(pred_action, (255, 0, 255))
    cv2.putText(img, "green=true  magenta=pred", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"true(lin={true_action[0]:.2f},ang={true_action[1]:.2f})", (8, panel - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 255, 150), 1, cv2.LINE_AA)
    cv2.putText(img, f"pred(lin={pred_action[0]:.2f},ang={pred_action[1]:.2f})", (8, panel - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 150, 255), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------
# 4) 메인
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 데이터셋 생성
    ap.add_argument("--rides", nargs="*", default=None,
                     help="ride 폴더 경로/glob 패턴들 (생략 시 data/ 아래 ride_* 전체 자동 탐색)")
    ap.add_argument("--estimator", choices=["heuristic", "sam2"], default="heuristic",
                     help="데이터셋 생성에 쓸 traversability 추정기. heuristic=기본(빠름), "
                          "sam2=정확하지만 훨씬 느림(GPU 권장, --frame-stride를 키워서 쓰길 권장)")
    ap.add_argument("--sam2-model", default="facebook/sam2.1-hiera-tiny")
    ap.add_argument("--frame-stride", type=int, default=20, help="ride당 프레임 샘플링 간격")
    ap.add_argument("--max-frames-per-ride", type=int, default=250, help="ride당 최대 샘플 수")
    ap.add_argument("--cam-height", type=float, default=0.25)
    ap.add_argument("--hfov", type=float, default=110.0)
    ap.add_argument("--cam-pitch-deg", type=float, default=0.0)
    ap.add_argument("--bev-range", type=float, default=8.0)
    ap.add_argument("--bev-resolution", type=float, default=0.05)
    ap.add_argument("--horizon-ratio", type=float, default=0.45)
    ap.add_argument("--bev-fill-max-gap", type=float, default=1.5)
    ap.add_argument("--no-bev-fill", action="store_true")
    ap.add_argument("--cache", default=CACHE_DEFAULT, help="BEV 데이터셋 캐시(.npz) 경로")
    ap.add_argument("--rebuild", action="store_true", help="캐시 무시하고 데이터셋을 다시 생성")
    # 학습
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-ang", type=float, default=3.0, help="angular 손실 가중치 (03b와 동일 취지)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out-dir", default=RUNS_DIR, help="체크포인트/그래프 저장 폴더")
    ap.add_argument("--num-qual-examples", type=int, default=8,
                     help="정성 평가(화살표 시각화)에 사용할 검증셋 예시 개수")
    args = ap.parse_args()

    X, y, ride, idx = load_or_build_dataset(args)
    print(f"[data] 총 {len(X)}개 샘플, BEV grid shape {X.shape[1:]}")

    tr_idx, va_idx = time_ordered_split(ride, args.val_frac)
    print(f"[split] train {len(tr_idx)} / val {len(va_idx)} "
          f"(ride별 마지막 {args.val_frac * 100:.0f}% 시간순 홀드아웃)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    tr_loader = DataLoader(BevGridDataset(X[tr_idx], y[tr_idx]), batch_size=args.bs, shuffle=True)
    va_loader = DataLoader(BevGridDataset(X[va_idx], y[va_idx]), batch_size=args.bs, shuffle=False)

    model = BevPolicyNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "best_bev_policy.pt")

    hist = {"tr": [], "va": []}
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        trl, trm = run_epoch(model, tr_loader, args.w_ang, device, opt)
        val, vam = run_epoch(model, va_loader, args.w_ang, device, None)
        sched.step()
        hist["tr"].append(trl)
        hist["va"].append(val)
        print(f"ep{ep:02d} | train {trl:.4f} | val {val:.4f} "
              f"| val MAE lin {vam[0]:.3f} ang {vam[1]:.3f}")
        if val < best:
            best = val
            torch.save(model.state_dict(), ckpt_path)
    print(f"[done] best val loss {best:.4f} -> {ckpt_path}")

    # 손실 곡선 + 검증셋 예측 vs 정답 산점도 (03b_train_bc.py와 동일한 형식)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(hist["tr"], label="train")
    plt.plot(hist["va"], label="val")
    plt.xlabel("epoch"); plt.ylabel("weighted MSE"); plt.legend()
    plt.title("BEV policy training")
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=120)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for x, y_batch in va_loader:
            P.append(model(x.to(device)).cpu().numpy())
            Y.append(y_batch.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for k, name in enumerate(["linear", "angular"]):
        ax[k].scatter(Y[:, k], P[:, k], s=5, alpha=.3)
        lim = [-1.1, 1.1]
        ax[k].plot(lim, lim, "r--")
        ax[k].set_xlim(lim); ax[k].set_ylim(lim)
        ax[k].set_xlabel("true"); ax[k].set_ylabel("pred"); ax[k].set_title(name)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "val_predictions.png"), dpi=120)
    print(f"[plot] {args.out_dir}/loss_curve.png, val_predictions.png")

    # 정성 평가: BEV + 정답/예측 화살표를 debug/bev_policy_eval/ 에 저장
    qual_dir = os.path.join(DEBUG_DIR, "bev_policy_eval")
    os.makedirs(qual_dir, exist_ok=True)
    n_show = min(args.num_qual_examples, len(va_idx))
    chosen = np.linspace(0, len(va_idx) - 1, n_show, dtype=int) if n_show > 0 else []
    for k in chosen:
        i = va_idx[k]
        with torch.no_grad():
            pred_action = model(
                torch.from_numpy(X[i]).unsqueeze(0).unsqueeze(0).to(device)
            ).cpu().numpy()[0]
        panel = render_policy_eval(X[i], y[i], pred_action, bev_range_m=args.bev_range)
        out_path = os.path.join(qual_dir, f"ride_{ride[i]}_frame_{idx[i]:06d}.png")
        cv2.imwrite(out_path, panel)
    print(f"[qual] {len(chosen)}장 -> {qual_dir}")


if __name__ == "__main__":
    main()
