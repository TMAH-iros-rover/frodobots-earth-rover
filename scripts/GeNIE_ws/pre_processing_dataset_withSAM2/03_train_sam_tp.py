"""
scripts/GeNIE_ws/pre_processing_dataset_withSAM2/03_train_sam_tp.py
02_annotate_app.py로 만든 라벨(data/labels/*.png)을 가지고 SAM2를 "SAM-TP"
(GeNIE 논문 Sec III-B)처럼 traversability 전용 모델로 파인튜닝한다.

    (01+02가 만든) 이미지+최종 traversability mask 쌍들 -> [이 스크립트로 SAM2 파인튜닝] -> SAM-TP 체크포인트

════════════════════════════════════════════════════════════════════════════
 "학습 가능한 prompt token"을 어떻게 구현했나 (중요)
════════════════════════════════════════════════════════════════════════════
논문은 "prompt encoder를 제거하고 'traversable'이라는 개념을 담은 학습 가능한
prompt token 1개로 대체했다"고 설명한다(Sec III-B). 처음엔 이걸 직접 구현하려고
Sam2Model 내부(prompt_encoder/mask_decoder)를 직접 호출하는 방식을 고려했는데,
HuggingFace transformers의 Sam2PromptEncoder 코드를 뜯어보니 **이미 정확히 이
역할을 하는 파라미터가 내장돼 있었다**:

  - point/box를 하나도 안 주고 Sam2Model을 호출하면(Sam2Model.forward 내부),
    "빈 점 1개(좌표 (0,0), label=-1)"로 자동 패딩한다.
  - Sam2PromptEncoder._embed_points()는 label=-1인 점의 임베딩을
    `self.not_a_point_embed.weight` (nn.Embedding(1, hidden_size), 즉 학습 가능한
    파라미터 1개)로 그대로 대체해버린다. 좌표 기반 위치 임베딩은 계산됐다가
    버려진다.
  - mask 입력도 안 주면 dense embedding은 `self.no_mask_embed.weight`
    (역시 학습 가능한 파라미터 1개)를 공간 전체에 복제해서 사용한다.

즉 "이미지만 주고 point/box/mask를 전혀 안 주는 채로 SAM2를 호출"하면, 그 자체로
이미 "학습 가능한 단일 prompt token(not_a_point_embed + no_mask_embed)"을 쓰는
구조가 된다 — 논문이 설명한 메커니즘과 사실상 동일하다. 그래서 이 스크립트는
Sam2Model을 전혀 수정하지 않고, 그냥 `model(pixel_values=..., multimask_output=False)`
를 point/box 없이 호출해서 나온 마스크 로짓에 traversability 라벨로 loss를 걸어
역전파하는 표준적인 파인튜닝 루프로 구현했다. (별도로 `python3 -c` 스모크 테스트로
not_a_point_embed / mask_decoder 파라미터에 실제로 gradient가 흐르고, 동결한
vision_encoder에는 안 흐르는 것까지 확인했다.)

════════════════════════════════════════════════════════════════════════════
 데이터 규모에 대한 솔직한 주의사항
════════════════════════════════════════════════════════════════════════════
논문은 15,347장으로 학습했다. 지금 우리 라벨은 (02번 스크립트로 만든) 수십 장
수준이다 — 이 정도로는 논문의 "Ours"(이미지 인코더 전체 + 디코더 전체 파인튜닝)
설정을 따라 하면 거의 확실히 과적합한다. 그래서 기본값은 논문의 D1/D2 계열에
가깝게, **이미지 인코더(vision_encoder, 백본)는 동결**하고 prompt_encoder(정확히는
not_a_point_embed/no_mask_embed) + mask_decoder만 파인튜닝한다
(`--unfreeze-encoder`로 바꿀 수 있지만, 데이터가 훨씬 더 많아지기 전엔 권장 안 함).
학습 후 반드시 val loss/IoU 곡선과 `runs/sam_tp/eval/`의 정성 결과를 직접 보고
과적합 여부를 판단할 것.

출력물:
  - runs/sam_tp/best_sam_tp.pt        : val loss 기준 최고 체크포인트 (model.state_dict())
  - runs/sam_tp/loss_curve.png        : train/val loss, train/val IoU 곡선
  - runs/sam_tp/eval/<sample_id>.png  : val 샘플에 대한 [원본 | 정답 | 예측] 비교 이미지

사용:
  python3 scripts/GeNIE_ws/pre_processing_dataset_withSAM2/03_train_sam_tp.py --epochs 60
필요 패키지: torch, transformers(Sam2Model 포함), opencv-python, numpy, matplotlib
"""
import os
import csv
import argparse
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Sam2Model, Sam2Processor

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS_DIR)))  # .../GeNIE_ws/pre_processing_dataset_withSAM2 -> GeNIE_ws -> scripts -> repo root
DATA_DIR = os.path.join(THIS_DIR, "data")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")
RUNS_DIR = os.path.join(REPO_ROOT, "runs", "sam_tp")


# --------------------------------------------------------------------------
# 1) 데이터
# --------------------------------------------------------------------------

def load_labeled_samples():
    """manifest.csv에서 annotated=1인 sample_id 목록 -> (frame_path, label_path) 리스트."""
    samples = []
    with open(MANIFEST_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["annotated"] != "1":
                continue
            sid = row["sample_id"]
            fp = os.path.join(FRAMES_DIR, f"{sid}.jpg")
            lp = os.path.join(LABELS_DIR, f"{sid}.png")
            if os.path.isfile(fp) and os.path.isfile(lp):
                samples.append((sid, fp, lp))
    return samples


def split_samples(samples, val_frac, seed):
    rng = random.Random(seed)
    samples = samples[:]
    rng.shuffle(samples)
    n_val = max(1, int(round(len(samples) * val_frac)))
    return samples[n_val:], samples[:n_val]


# --------------------------------------------------------------------------
# 2) 손실/평가 지표
# --------------------------------------------------------------------------

def bce_dice_loss(logits, target, eps=1e-6):
    """logits, target: (B, H, W). BCE + Dice 조합 (표준 세그멘테이션 손실)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits).flatten(1)
    tgt = target.flatten(1)
    inter = (probs * tgt).sum(1)
    union = probs.sum(1) + tgt.sum(1)
    dice = 1 - (2 * inter + eps) / (union + eps)
    return bce + dice.mean(), bce.item(), dice.mean().item()


def iou_at_thresh(logits, target, thresh=0.5, eps=1e-6):
    pred = (torch.sigmoid(logits) > thresh).float().flatten(1)
    tgt = target.flatten(1)
    inter = (pred * tgt).sum(1)
    union = ((pred + tgt) > 0).float().sum(1)
    return ((inter + eps) / (union + eps)).mean().item()


# --------------------------------------------------------------------------
# 3) forward: 프롬프트 없이 호출 (not_a_point_embed/no_mask_embed가 곧 "prompt token")
# --------------------------------------------------------------------------

def forward_no_prompt(model, processor, frame_bgr, device):
    """이미지 1장 -> (low_res_logits (1,h,w), (h,w)). point/box/mask를 하나도 안 줘서
    Sam2Model이 자동으로 학습 가능한 not_a_point_embed/no_mask_embed를 prompt로 쓰게 한다."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=frame_rgb, return_tensors="pt").to(device)
    out = model(**inputs, multimask_output=False)
    logits = out.pred_masks[:, 0, 0]  # (1, h_low, w_low)
    return logits


def load_gt_lowres(label_path, size_hw, device):
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(label, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy((gt > 0).astype(np.float32))[None].to(device)


# --------------------------------------------------------------------------
# 4) 메인
# --------------------------------------------------------------------------

def run_epoch(model, processor, samples, device, opt, bs):
    """opt가 주어지면 학습(그래디언트 누적 bs개씩 모아 step), None이면 검증만."""
    train = opt is not None
    model.train(train)
    torch.set_grad_enabled(train)

    tot_loss, tot_bce, tot_dice, tot_iou, n = 0.0, 0.0, 0.0, 0.0, 0
    order = list(range(len(samples)))
    if train:
        random.shuffle(order)

    if train:
        opt.zero_grad()
    accum = 0
    for count, idx in enumerate(order, 1):
        sid, fp, lp = samples[idx]
        frame = cv2.imread(fp)
        logits = forward_no_prompt(model, processor, frame, device)
        gt = load_gt_lowres(lp, logits.shape[-2:], device)
        loss, bce_v, dice_v = bce_dice_loss(logits, gt)
        iou = iou_at_thresh(logits.detach(), gt)

        if train:
            (loss / bs).backward()
            accum += 1
            if accum == bs or count == len(order):
                opt.step()
                opt.zero_grad()
                accum = 0

        tot_loss += loss.item(); tot_bce += bce_v; tot_dice += dice_v; tot_iou += iou; n += 1

    return dict(loss=tot_loss / n, bce=tot_bce / n, dice=tot_dice / n, iou=tot_iou / n)


def save_qualitative(model, processor, samples, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for sid, fp, lp in samples:
            frame = cv2.imread(fp)
            h, w = frame.shape[:2]
            logits = forward_no_prompt(model, processor, frame, device)
            prob = torch.sigmoid(logits)[0]
            prob_up = F.interpolate(prob[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            pred = (prob_up > 0.5).cpu().numpy()
            gt = cv2.imread(lp, cv2.IMREAD_GRAYSCALE) > 0

            gt_panel = frame.copy()
            gt_panel[gt] = (gt_panel[gt] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
            pred_panel = frame.copy()
            pred_panel[pred] = (pred_panel[pred] * 0.5 + np.array([255, 0, 255]) * 0.5).astype(np.uint8)
            cv2.putText(gt_panel, "GT (label)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(pred_panel, "SAM-TP pred", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            sep = np.full((h, 6, 3), 60, dtype=np.uint8)
            panel = np.hstack([frame, sep, gt_panel, sep, pred_panel])
            cv2.imwrite(os.path.join(out_dir, f"{sid}.png"), panel)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sam2-model", default="facebook/sam2.1-hiera-tiny")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=4, help="그래디언트 누적 배치 크기 (이미지 1장씩 forward 후 누적)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--unfreeze-encoder", action="store_true",
                     help="vision_encoder(백본)까지 파인튜닝 (기본은 동결 -> 데이터가 적어서 권장 안 함)")
    ap.add_argument("--out-dir", default=RUNS_DIR)
    args = ap.parse_args()

    samples = load_labeled_samples()
    if len(samples) < 4:
        raise SystemExit(f"라벨링된 샘플이 너무 적음({len(samples)}개). 02_annotate_app.py로 더 라벨링해줘.")
    train_samples, val_samples = split_samples(samples, args.val_frac, args.seed)
    print(f"[data] 전체 {len(samples)}개 -> train {len(train_samples)} / val {len(val_samples)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    processor = Sam2Processor.from_pretrained(args.sam2_model)
    model = Sam2Model.from_pretrained(args.sam2_model).to(device)

    if not args.unfreeze_encoder:
        for p in model.vision_encoder.parameters():
            p.requires_grad_(False)
        print("[freeze] vision_encoder 동결 (prompt_encoder + mask_decoder만 학습)")
    else:
        print("[freeze] 전체 파인튜닝 (vision_encoder 포함) -- 데이터가 적으면 과적합 위험 큼")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[params] 학습 대상 파라미터 {n_trainable:,}개")
    opt = torch.optim.Adam(trainable, lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "best_sam_tp.pt")

    hist = {"tr_loss": [], "va_loss": [], "tr_iou": [], "va_iou": []}
    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, processor, train_samples, device, opt, args.bs)
        va = run_epoch(model, processor, val_samples, device, None, args.bs)
        hist["tr_loss"].append(tr["loss"]); hist["va_loss"].append(va["loss"])
        hist["tr_iou"].append(tr["iou"]); hist["va_iou"].append(va["iou"])
        print(f"ep{ep:03d} | train loss {tr['loss']:.4f} iou {tr['iou']:.3f} "
              f"| val loss {va['loss']:.4f} iou {va['iou']:.3f}")
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(model.state_dict(), ckpt_path)

    print(f"[done] best val loss {best_val:.4f} -> {ckpt_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(hist["tr_loss"], label="train"); ax[0].plot(hist["va_loss"], label="val")
    ax[0].set_title("loss (BCE+Dice)"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["tr_iou"], label="train"); ax[1].plot(hist["va_iou"], label="val")
    ax[1].set_title("IoU@0.5"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=120)
    print(f"[plot] {args.out_dir}/loss_curve.png")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    save_qualitative(model, processor, val_samples, device, os.path.join(args.out_dir, "eval"))
    print(f"[qual] val {len(val_samples)}장 -> {args.out_dir}/eval/")

    if hist["va_loss"][-1] < hist["tr_loss"][-1] * 0.3 or hist["tr_iou"][-1] - hist["va_iou"][-1] > 0.25:
        print("\n[warn] train/val 성능 차이가 큽니다 -- 과적합 가능성이 높습니다. "
              "라벨을 더 모으거나(01번 스크립트 재실행), --unfreeze-encoder 없이(기본값) "
              "더 강하게 규제하는 걸 권장합니다.")


if __name__ == "__main__":
    main()
