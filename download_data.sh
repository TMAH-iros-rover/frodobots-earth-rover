#!/usr/bin/env bash
# download_data.sh
# FrodoBots-2K getting-started 샘플(세션 2개)을 data/ 에 받아서 압축 해제.
# 데이터는 repo에 포함하지 않으므로(라이선스 CC-BY-SA-4.0), 각자 이 스크립트로 받는다.
#
# 사용:
#   bash scripts/download_data.sh
#   (또는  chmod +x scripts/download_data.sh && ./scripts/download_data.sh )

set -e  # 에러 나면 즉시 중단

# repo 루트 기준 data/ 로 이동 (스크립트가 scripts/ 안에 있어도 동작)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$REPO_ROOT/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

URL="https://frodobots-2k-dataset.s3.ap-southeast-1.amazonaws.com/frodobots-dataset-getting-started.zip"
ZIP="frodobots-dataset-getting-started.zip"

# 이미 풀려 있으면 스킵
if [ -d "frodobots-dataset-getting-started" ]; then
  echo "[skip] 이미 존재: $DATA_DIR/frodobots-dataset-getting-started"
  exit 0
fi

echo "[download] $URL"
if command -v wget >/dev/null 2>&1; then
  wget -c "$URL" -O "$ZIP"
else
  curl -L -C - "$URL" -o "$ZIP"
fi

echo "[unzip] $ZIP"
if ! command -v unzip >/dev/null 2>&1; then
  echo "[err] unzip 없음.  sudo apt install unzip -y  후 다시 실행"
  exit 1
fi
unzip -q "$ZIP"
rm -f "$ZIP"                    # 용량 절약 (원하면 이 줄 지우면 zip 유지)
rm -rf __MACOSX                 # 맥 잔여물 정리

echo ""
echo "[done] 다운로드 완료. 세션 목록:"
ls -d frodobots-dataset-getting-started/ride_* 2>/dev/null

echo ""
echo "다음 단계:"
echo "  python scripts/02_inspect_session.py data/frodobots-dataset-getting-started/ride_19154_20240225023555"