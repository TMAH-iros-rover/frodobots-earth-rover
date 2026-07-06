"""
03b_train_bc.py
카메라 이미지 -> 제어(linear, angular) 를 예측하는 behavior cloning 베이스라인.
ResNet18(사전학습) 백본 + 2출력 회귀 head, 출력은 tanh 로 [-1,1] 로 bound.

사용:
  pip install torch torchvision   # Blackwell(50xx)면 아래 '설치 주의' 참고
  python 03b_train_bc.py --manifest train_data/manifest.csv --epochs 15

핵심 설계
  - 타깃: [linear, angular]  (게이머 입력, 둘 다 -1~1)
  - 검증분할: 마지막 20% '시간순' 홀드아웃 (인접 프레임 유사 -> 랜덤분할은 낙관적)
  - 손실: MSE. angular 가 중요한데 값이 작아 가중치(--w_ang) 부여 가능
"""
import os, argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DrivingDataset(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
        aug = [transforms.ColorJitter(0.2, 0.2, 0.2)] if train else []
        self.tf = transforms.Compose(
            aug + [transforms.ToTensor(),
                   transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.frame_path).convert("RGB")
        x = self.tf(img)
        y = torch.tensor([row.linear, row.angular], dtype=torch.float32)
        return x, y


def build_model():
    try:
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception as e:
        print(f"[warn] 사전학습 가중치 다운로드 실패({e}) -> 랜덤 초기화로 진행")
        m = resnet18(weights=None)
    m.fc = nn.Sequential(nn.Linear(512, 2), nn.Tanh())  # 출력 [-1,1]
    return m


def run_epoch(model, loader, loss_fn, w, device, opt=None):
    train = opt is not None
    model.train(train)
    tot, mae = 0.0, np.zeros(2)
    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        per = ((pred - y) ** 2).mean(0)            # [lin, ang] 별 MSE
        loss = per[0] + w * per[1]
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        bs = x.size(0)
        tot += loss.item() * bs
        mae += (pred - y).abs().mean(0).detach().cpu().numpy() * bs
    n = len(loader.dataset)
    return tot / n, mae / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="train_data/manifest.csv")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w_ang", type=float, default=3.0, help="angular 손실 가중치")
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    df = pd.read_csv(args.manifest)
    split = int(len(df) * (1 - args.val_frac))
    tr_df, va_df = df.iloc[:split], df.iloc[split:]          # 시간순 홀드아웃
    print(f"[data] train {len(tr_df)} / val {len(va_df)}")

    tr = DataLoader(DrivingDataset(tr_df, True), args.bs, shuffle=True,
                    num_workers=4, pin_memory=(device == "cuda"))
    va = DataLoader(DrivingDataset(va_df, False), args.bs, shuffle=False,
                    num_workers=4, pin_memory=(device == "cuda"))

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = None

    hist = {"tr": [], "va": []}
    best = 1e9
    for ep in range(1, args.epochs + 1):
        trl, trm = run_epoch(model, tr, loss_fn, args.w_ang, device, opt)
        val, vam = run_epoch(model, va, loss_fn, args.w_ang, device, None)
        sched.step()
        hist["tr"].append(trl); hist["va"].append(val)
        print(f"ep{ep:02d} | train {trl:.4f} | val {val:.4f} "
              f"| val MAE lin {vam[0]:.3f} ang {vam[1]:.3f}")
        if val < best:
            best = val
            torch.save(model.state_dict(), "best_bc.pt")

    print(f"[done] best val loss {best:.4f} -> best_bc.pt")

    # 손실 곡선
    plt.figure()
    plt.plot(hist["tr"], label="train"); plt.plot(hist["va"], label="val")
    plt.xlabel("epoch"); plt.ylabel("weighted MSE"); plt.legend()
    plt.title("BC training"); plt.savefig("loss_curve.png", dpi=120)

    # 검증셋 예측 vs 정답 산점도
    model.load_state_dict(torch.load("best_bc.pt", map_location=device))
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for x, y in va:
            P.append(model(x.to(device)).cpu().numpy()); Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for k, name in enumerate(["linear", "angular"]):
        ax[k].scatter(Y[:, k], P[:, k], s=5, alpha=.3)
        lim = [-1.1, 1.1]; ax[k].plot(lim, lim, "r--")
        ax[k].set_xlim(lim); ax[k].set_ylim(lim)
        ax[k].set_xlabel("true"); ax[k].set_ylabel("pred"); ax[k].set_title(name)
    fig.tight_layout(); fig.savefig("val_predictions.png", dpi=120)
    print("[plot] loss_curve.png, val_predictions.png")


if __name__ == "__main__":
    main()
