#!/usr/bin/env bash
# labeling.sh
# minirover-<ID>.tar (예: minirover-0011.tar, minirover-0012.tar, ...) 하나를 받아서
#   1) 아직 안 풀려있으면 압축 해제 (data/minirover-<ID>/)
#   2) 01_generate_mask_proposals.py 로 그 데이터셋의 ride들만 대상으로 SAM2 mask 후보 생성
#   3) 02_annotate_app.py (라벨링 웹앱, http://localhost:5050) 실행
#
# 라벨링 결과(data/frames, data/proposals, data/labels, data/manifest.csv)는
# scripts/GeNIE_ws/pre_processing_dataset_withSAM2/data/ 밑에 기존 0011과 완전히 같은 방식으로
# "누적" 저장된다 (sample_id가 ride id 기반이라 데이터셋 여러 개를 계속 실행해도 안 섞이고
# 계속 쌓인다 -- 이미 라벨링된 프레임은 자동으로 건너뜀).
#
# 사용:
#   ./labeling.sh 0011            # data/minirover-0011 대상, 새 프레임 50장 후보 생성 후 라벨링 앱 실행
#   ./labeling.sh 0012 80         # 프레임 수를 80장으로 바꾸고 싶을 때
#
# minirover-<ID>.tar 파일은 이 스크립트와 같은 repo 루트에 미리 받아둬야 한다.

set -e

if [ -z "$1" ]; then
  echo "사용법: ./labeling.sh <데이터셋 번호(예: 0011)> [생성할 프레임 수, 기본 50]"
  exit 1
fi

ID="$1"
NUM_FRAMES="${2:-50}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

TAR_PATH="$REPO_ROOT/minirover-${ID}.tar"
DATASET_DIR="$REPO_ROOT/data/minirover-${ID}"
PREP_DIR="$REPO_ROOT/scripts/GeNIE_ws/pre_processing_dataset_withSAM2"

SAM2_PY="/home/kante/miniconda3/envs/isaaclab51/bin/python3"   # torch + transformers(Sam2Model), GPU
ANNOTATE_PY="python3"                                          # base python: flask + opencv-python + numpy

# 1) 압축 해제 (이미 풀려있으면 스킵)
if [ -d "$DATASET_DIR" ] && [ -n "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]; then
  echo "[skip] 이미 압축 해제됨: $DATASET_DIR"
else
  if [ ! -f "$TAR_PATH" ]; then
    echo "[err] $TAR_PATH 를 찾을 수 없음. repo 루트에 minirover-${ID}.tar 를 먼저 받아둬."
    exit 1
  fi
  echo "[extract] $TAR_PATH -> $DATASET_DIR"
  mkdir -p "$DATASET_DIR"
  tar -xf "$TAR_PATH" -C "$DATASET_DIR"
fi

# 2) SAM2 mask 후보 생성 (이 데이터셋의 ride들만 대상, 이미 처리된 프레임은 01번 스크립트가 자동으로 건너뜀)
echo ""
echo "[step 1/2] SAM2 mask 후보 생성 (ride: data/minirover-${ID}/ride_*, 목표 프레임 ${NUM_FRAMES}장)"
"$SAM2_PY" "$PREP_DIR/01_generate_mask_proposals.py" \
  --rides "$DATASET_DIR"/ride_* \
  --num-frames "$NUM_FRAMES"

# 3) 라벨링 웹앱 실행
echo ""
echo "[step 2/2] 라벨링 웹앱 실행 -> http://localhost:5050"
echo "  (원격 서버라면 다른 터미널에서: ssh -L 5050:localhost:5050 <host> 포트포워딩 후 로컬 브라우저 접속)"
echo "  Ctrl+C 로 종료. 미완료(annotated=0) 프레임부터 이어서 라벨링된다."
echo ""
"$ANNOTATE_PY" "$PREP_DIR/02_annotate_app.py"
