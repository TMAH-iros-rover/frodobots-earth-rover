"""
01_download_session.py
FrodoBots-2K 에서 '한 세션만' 받아서 빠르게 테스트하기 위한 스크립트.

동작:
  1) HF 데이터셋 repo 파일 목록을 확인
  2) 다운로드 링크가 담긴 csv (complete_dataset.csv / complete-dataset.csv 등) 자동 탐지 후 받기
  3) csv의 첫 번째 url(zip) 하나만 받아서 ./frodo_data/ 에 압축 해제
  4) 안에 있는 ride_* 세션 폴더 경로를 출력

사용:
  pip install huggingface_hub pandas requests
  python 01_download_session.py
"""
import os
import zipfile
import glob
import pandas as pd
import requests
from huggingface_hub import list_repo_files, hf_hub_download

REPO_ID = "BitRobot/FrodoBots-2K"        # 구경로: "frodobots/FrodoBots-2K"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(REPO_ROOT, "data", "frodobots-2k-full")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # 1) repo 파일 목록
    files = list_repo_files(REPO_ID, repo_type="dataset")
    print(f"[repo] {REPO_ID} 파일 {len(files)}개")
    for f in files[:30]:
        print("   ", f)

    # 2) 다운로드 링크 csv 자동 탐지
    csv_candidates = [f for f in files
                      if f.lower().endswith(".csv") and "dataset" in f.lower()]
    if not csv_candidates:
        csv_candidates = [f for f in files if f.lower().endswith(".csv")]
    assert csv_candidates, "csv 파일을 못 찾음. repo 파일 목록을 직접 확인해줘."
    csv_name = csv_candidates[0]
    print(f"\n[csv] 사용할 링크 csv: {csv_name}")

    csv_path = hf_hub_download(REPO_ID, csv_name, repo_type="dataset",
                               local_dir=OUT_DIR)
    df = pd.read_csv(csv_path)
    print(f"[csv] 총 {len(df)}개 세션 링크. 컬럼: {list(df.columns)}")

    # url 컬럼 자동 탐지
    url_col = "url" if "url" in df.columns else \
        next((c for c in df.columns if df[c].astype(str).str.startswith("http").any()), None)
    assert url_col, "url 컬럼을 못 찾음. df.head() 로 직접 확인해줘."

    # 3) 첫 세션 zip 하나만 받기
    url = df[url_col].iloc[0]
    zip_name = url.split("/")[-1].split("?")[0]
    zip_path = os.path.join(OUT_DIR, zip_name)
    print(f"\n[download] {url}\n        -> {zip_path}")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(zip_path, "wb") as fp:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fp.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r   {done/1e6:7.1f} / {total/1e6:7.1f} MB", end="")
    print("\n[download] 완료")

    # 4) 압축 해제
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(OUT_DIR)
    print("[unzip] 완료")

    rides = sorted(glob.glob(os.path.join(OUT_DIR, "**", "ride_*"), recursive=True))
    rides = [r for r in rides if os.path.isdir(r)]
    print(f"\n[done] 세션 폴더 {len(rides)}개:")
    for r in rides[:5]:
        print("   ", r)
    if rides:
        print(f"\n다음 단계:  python 02_inspect_session.py \"{rides[0]}\"")


if __name__ == "__main__":
    main()
