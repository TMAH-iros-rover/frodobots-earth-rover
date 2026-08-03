#!/usr/bin/env bash
# training.sh
# 지금까지 labeling.sh 로 쌓아온 라벨(scripts/GeNIE_ws/pre_processing_dataset_withSAM2/data/labels/*.png,
# manifest.csv의 annotated=1 전체 -- 특정 데이터셋 번호에 한정되지 않고 "누적된 전체 라벨"을 대상으로 함,
# 03_train_sam_tp.py의 load_labeled_samples()가 manifest.csv 전체에서 annotated==1인 것만 걸러서 씀)
# 로 SAM-TP(SAM2 파인튜닝)를 학습한다.
#
# 사용:
#   ./training.sh              # 기본 60 epoch
#   ./training.sh 100          # epoch 수 직접 지정
#
# 결과: runs/sam_tp/best_sam_tp.pt, runs/sam_tp/loss_curve.png, runs/sam_tp/eval/*.png

set -e

EPOCHS="${1:-60}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

PREP_DIR="$REPO_ROOT/scripts/GeNIE_ws/pre_processing_dataset_withSAM2"
SAM2_PY="/home/kante/miniconda3/envs/isaaclab51/bin/python3"   # torch + transformers(Sam2Model) + matplotlib, GPU

echo "[train] 누적된 전체 라벨 데이터로 SAM-TP 파인튜닝 (epochs=${EPOCHS})"
"$SAM2_PY" "$PREP_DIR/03_train_sam_tp.py" --epochs "$EPOCHS"
