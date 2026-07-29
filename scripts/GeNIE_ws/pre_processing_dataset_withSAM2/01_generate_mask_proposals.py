"""
scripts/GeNIE_ws/pre_processing_dataset_withSAM2/01_generate_mask_proposals.py
GeNIE 논문(Sec III-B)의 반자동 라벨링 파이프라인 1단계를 재현한다:

    원본 이미지 -> [SAM2로 여러 영역 mask 제안] -> (사람이 선택/수정, 02번 스크립트) -> 최종 traversability mask

이 스크립트는 그중 "SAM2로 여러 영역 mask 제안" 부분을 담당한다. 논문/원본 SAM2
저장소의 SAM2AutomaticMaskGenerator는 이미지 전체에 격자 모양으로 점을 뿌려서
점마다 "이 점을 포함하는 영역이 뭐야?"를 SAM2에 물어보고, 겹치는 결과를 병합/
중복제거해서 서로 다른 영역 후보들을 만든다. 여기서는 HuggingFace transformers의
Sam2Model로 같은 아이디어를 직접 구현했다:

  1) 이미지 위에 grid_nx x grid_ny 격자로 점을 뿌린다 (각 점 = 독립적인 1점 프롬프트).
  2) 이미지 인코더(제일 비싼 연산)는 프레임당 딱 한 번만 돌리고, 모든 격자점을
     하나의 배치(batch)로 묶어서 디코더만 여러 번 돌려 각 점에 대한 최상위(예측
     IoU가 가장 높은) 후보 마스크를 얻는다 -> 점 개수를 늘려도 비용이 거의 안 늘어남.
  3) 서로 IoU가 --iou-merge-thresh 이상 겹치는 마스크는 점수가 더 높은 것만 남기고
     중복 제거한다(NMS와 동일한 방식) -> 사실상 "같은 영역을 가리키는 여러 점"을
     하나의 제안으로 합치는 효과.
  4) 남은 후보를 점수 순으로 최대 --max-proposals 개까지, 04번 스크립트의
     find_ride_dirs()/extract_frames_at_indices() 등을 그대로 재사용해 data/ 안의
     ride들에서 프레임을 뽑아 처리한다.

각 프레임 결과는 다음 3가지로 저장된다 (02_annotate_app.py가 그대로 읽어서 씀):
  - data/frames/<sample_id>.jpg      : 원본 프레임 (작업 해상도로 리사이즈)
  - data/proposals/<sample_id>.npz   : masks(K,H,W bool, packbits로 압축), scores(K,)
  - data/manifest.csv                : sample_id, ride, frame_idx, w, h, num_proposals, annotated

사용:
  python3 scripts/GeNIE_ws/pre_processing_dataset_withSAM2/01_generate_mask_proposals.py --num-frames 50
  (필요 패키지: torch, transformers(Sam2Model 포함), opencv-python, numpy)
"""
import os
import sys
import csv
import argparse
import importlib.util

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Sam2Model, Sam2Processor

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# scripts/GeNIE_ws/pre_processing_dataset_withSAM2 -> GeNIE_ws -> scripts -> repo root (3단계 위)
SCRIPTS_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
DATA_OUT = os.path.join(THIS_DIR, "data")
FRAMES_DIR = os.path.join(DATA_OUT, "frames")
PROPOSALS_DIR = os.path.join(DATA_OUT, "proposals")
MANIFEST_PATH = os.path.join(DATA_OUT, "manifest.csv")


def _load_bev_module():
    """04_bev_traversability.py의 세션 탐색/프레임 추출 유틸을 재사용 (파일명이
    숫자로 시작해서 일반 import 대신 importlib로 직접 로드)."""
    path = os.path.join(SCRIPTS_DIR, "04_bev_traversability.py")
    spec = importlib.util.spec_from_file_location("bev_traversability", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bevmod = _load_bev_module()


# --------------------------------------------------------------------------
# 1) 프레임 선정: data/ 아래 ride들에서 --num-frames 장을 고르게 분배해서 샘플링
# --------------------------------------------------------------------------

def pick_samples(ride_dirs, num_frames, seed=0):
    """여러 ride에 num_frames 장을 최대한 고르게 분배해서 (session_dir, rid, frame_idx)
    목록을 만든다. 다양한 ride(시간대/장소가 다름)가 골고루 섞여야 나중에 만들
    traversability 데이터셋이 편향되지 않는다."""
    rng = np.random.default_rng(seed)
    per_ride = max(1, num_frames // max(1, len(ride_dirs)))
    samples = []
    for session_dir in ride_dirs:
        rid = bevmod.parse_ride_id(session_dir)
        total = bevmod.count_frames_hint(session_dir, rid)
        if not total:
            continue
        n = min(per_ride, total)
        # linspace로 고르게 뽑으면 n=1일 때 항상 프레임 0(주로 출발 지점, 장면이 비슷함)만
        # 뽑히므로, 무작위 비복원추출로 다양성을 확보한다.
        idxs = sorted(rng.choice(total, size=n, replace=False).tolist())
        for i in idxs:
            samples.append((session_dir, rid, int(i)))
    rng.shuffle(samples)
    return samples[:num_frames]


# --------------------------------------------------------------------------
# 2) SAM2 격자 프롬프트로 영역 후보 생성 (자체 구현 automatic mask generator)
# --------------------------------------------------------------------------

def sample_grid_points(w, h, nx, ny, margin_frac=0.06):
    xs = np.linspace(w * margin_frac, w * (1 - margin_frac), nx)
    ys = np.linspace(h * margin_frac, h * (1 - margin_frac), ny)
    return [[int(x), int(y)] for y in ys for x in xs]


def generate_proposals(model, processor, device, image_rgb, grid_nx=7, grid_ny=6,
                        iou_merge_thresh=0.85, max_proposals=20, min_area_frac=0.002,
                        mask_thresh=0.5):
    """image_rgb(HxWx3) 한 장 -> (masks: list[HxW bool], scores: list[float]).
    격자점마다 최상위 후보 마스크를 뽑은 뒤, 서로 많이 겹치는 것들은 점수가 더 높은
    것만 남기고 중복 제거(NMS)한다."""
    h, w = image_rgb.shape[:2]
    points = sample_grid_points(w, h, grid_nx, grid_ny)
    n = len(points)

    input_points = [[[p] for p in points]]  # 배치=이미지 1장, 프롬프트 n개, 프롬프트당 점 1개
    input_labels = [[[1] for _ in points]]

    inputs = processor(images=image_rgb, input_points=input_points, input_labels=input_labels,
                        return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)

    logits = out.pred_masks[0]         # (n, 3, h_low, w_low)
    ious = out.iou_scores[0]           # (n, 3)
    best_idx = ious.argmax(dim=1)
    ar = torch.arange(n, device=ious.device)
    best_logits = logits[ar, best_idx]     # (n, h_low, w_low)
    best_ious = ious[ar, best_idx]         # (n,)

    probs = torch.sigmoid(best_logits)
    probs_up = F.interpolate(probs.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False)[:, 0]
    masks = (probs_up > mask_thresh).cpu().numpy()
    scores = best_ious.float().cpu().numpy()

    min_area = min_area_frac * h * w
    order = np.argsort(-scores)
    kept_masks, kept_scores = [], []
    for idx in order:
        m = masks[idx]
        if m.sum() < min_area:
            continue
        dup = False
        for km in kept_masks:
            inter = np.logical_and(m, km).sum()
            union = np.logical_or(m, km).sum()
            if union > 0 and inter / union > iou_merge_thresh:
                dup = True
                break
        if not dup:
            kept_masks.append(m)
            kept_scores.append(float(scores[idx]))
        if len(kept_masks) >= max_proposals:
            break
    return kept_masks, kept_scores


# --------------------------------------------------------------------------
# 3) 메인
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rides", nargs="*", default=None, help="ride 폴더 경로/glob (생략 시 data/ 전체 자동 탐색)")
    ap.add_argument("--num-frames", type=int, default=50, help="이번에 새로 전처리할 프레임 수")
    ap.add_argument("--work-width", type=int, default=960, help="저장/작업용 리사이즈 너비[px]")
    ap.add_argument("--grid-nx", type=int, default=7)
    ap.add_argument("--grid-ny", type=int, default=6)
    ap.add_argument("--iou-merge-thresh", type=float, default=0.85)
    ap.add_argument("--max-proposals", type=int, default=20)
    ap.add_argument("--sam2-model", default="facebook/sam2.1-hiera-tiny")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(PROPOSALS_DIR, exist_ok=True)

    ride_dirs = args.rides
    if ride_dirs:
        import glob
        expanded = []
        for p in ride_dirs:
            expanded.extend(sorted(d for d in glob.glob(os.path.expanduser(p)) if os.path.isdir(d)))
        ride_dirs = expanded
    else:
        ride_dirs = bevmod.find_ride_dirs()
    if not ride_dirs:
        sys.exit("ride 폴더를 못 찾음. --rides 로 직접 지정해줘.")
    print(f"[rides] {len(ride_dirs)}개 후보에서 프레임 분배")

    # 이미 처리된 sample_id는 건너뛰기 위해 기존 manifest 확인
    existing_ids = set()
    if os.path.isfile(MANIFEST_PATH):
        with open(MANIFEST_PATH, newline="") as f:
            for row in csv.DictReader(f):
                existing_ids.add(row["sample_id"])

    samples = pick_samples(ride_dirs, args.num_frames, seed=args.seed)
    samples = [(s, r, i) for (s, r, i) in samples if f"{r}_f{i:06d}" not in existing_ids]
    print(f"[samples] 새로 처리할 프레임 {len(samples)}개 (기존 {len(existing_ids)}개는 건너뜀)")
    if not samples:
        print("[done] 새로 처리할 게 없음")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sam2] {args.sam2_model} 로딩 중 (device={device}) ...")
    processor = Sam2Processor.from_pretrained(args.sam2_model)
    model = Sam2Model.from_pretrained(args.sam2_model).to(device).eval()
    print("[sam2] 로딩 완료")

    # ride별로 모아서 처리 (같은 ride의 소스 파일 리스트/비디오 디코딩을 재사용하기 위함)
    by_ride = {}
    for session_dir, rid, frame_idx in samples:
        by_ride.setdefault((session_dir, rid), []).append(frame_idx)

    new_rows = []
    for (session_dir, rid), idxs in by_ride.items():
        sources, _ = bevmod.find_front_camera_sources(session_dir, rid)
        frames = bevmod.extract_frames_at_indices(sources, idxs)
        for frame_idx in idxs:
            if frame_idx not in frames:
                print(f"[skip] ride={rid} frame={frame_idx} 디코딩 실패")
                continue
            frame_bgr = frames[frame_idx]
            h0, w0 = frame_bgr.shape[:2]
            scale = args.work_width / w0
            work = cv2.resize(frame_bgr, (args.work_width, int(h0 * scale)))
            work_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)

            masks, scores = generate_proposals(
                model, processor, device, work_rgb,
                grid_nx=args.grid_nx, grid_ny=args.grid_ny,
                iou_merge_thresh=args.iou_merge_thresh, max_proposals=args.max_proposals,
            )

            sample_id = f"{rid}_f{frame_idx:06d}"
            cv2.imwrite(os.path.join(FRAMES_DIR, f"{sample_id}.jpg"), work, [cv2.IMWRITE_JPEG_QUALITY, 92])

            wh, ww = work.shape[:2]
            packed = np.packbits(np.stack(masks, axis=0).astype(np.uint8), axis=None) if masks else np.array([], dtype=np.uint8)
            np.savez_compressed(
                os.path.join(PROPOSALS_DIR, f"{sample_id}.npz"),
                packed=packed, k=len(masks), h=wh, w=ww,
                scores=np.asarray(scores, dtype=np.float32),
            )
            new_rows.append(dict(sample_id=sample_id, ride=rid, frame_idx=frame_idx,
                                  w=ww, h=wh, num_proposals=len(masks), annotated=0))
            print(f"[ok] {sample_id}: 후보 {len(masks)}개")

    header = ["sample_id", "ride", "frame_idx", "w", "h", "num_proposals", "annotated"]
    write_header = not os.path.isfile(MANIFEST_PATH)
    with open(MANIFEST_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        for row in new_rows:
            writer.writerow(row)

    print(f"\n[done] {len(new_rows)}개 프레임 전처리 완료 -> {MANIFEST_PATH}")
    print(f"다음 단계: python3 {os.path.join(THIS_DIR, '02_annotate_app.py')}")


if __name__ == "__main__":
    main()
