# 라벨링 / 학습 사용법

`minirover-<번호>.tar` (예: `minirover-0011.tar`, `minirover-0019.tar`, `minirover-0035.tar` ...)
데이터셋을 받아서 SAM-TP 라벨을 쌓고, 그걸로 SAM2를 파인튜닝하는 전체 과정 정리.

## 전체 흐름

```
minirover-<ID>.tar (repo 루트)
    -> [labeling.sh] 압축 해제 -> SAM2 mask 후보 생성 -> 웹 라벨링 앱
    -> (사람이 직접 라벨링)
    -> [training.sh] 지금까지 쌓인 라벨 전체로 SAM2 파인튜닝(SAM-TP)
```

## 1) 라벨링: `labeling.sh`

### 준비물
`minirover-<ID>.tar` 파일을 이 파일과 같은 repo 루트에 미리 받아둔다.
(예: `/home/kante/frodobots-earth-rover/minirover-0019.tar`)

### 실행

```bash
./labeling.sh 0019          # minirover-0019.tar 대상, 새 프레임 50장 후보 생성
./labeling.sh 0035 80       # 두 번째 인자로 프레임 수 지정 (기본 50)
```

### 실행하면 순서대로 벌어지는 일

1. **압축 해제** — `data/minirover-<ID>/`가 없으면 `minirover-<ID>.tar`를 풀어준다. 이미 풀려있으면 스킵.
2. **SAM2 mask 후보 생성** (`01_generate_mask_proposals.py`, GPU 사용, `isaaclab51` conda env로 실행) —
   그 데이터셋의 ride들에서 새 프레임을 골라 SAM2로 영역 후보를 만든다. 이미 처리된 프레임(다른
   데이터셋 포함)은 자동으로 건너뛴다. 이 단계는 프레임 수에 따라 시간이 좀 걸리고, 콘솔에
   `[ok] <sample_id>: 후보 N개` 로그가 프레임마다 찍힌다.
3. **라벨링 웹앱 실행** (`02_annotate_app.py`, Flask) — 끝나면 자동으로 뜬다. 콘솔에
   `Running on http://127.0.0.1:5050` 같은 로그가 보이면(`WARNING: This is a development server...`는
   Flask 표준 경고라 무시해도 됨) 정상적으로 뜬 것.

### 라벨링 방법 (브라우저)

브라우저에서 `http://localhost:5050` 접속 (원격 서버라면 로컬 터미널에서 먼저
`ssh -L 5050:localhost:5050 <host>` 포트포워딩 후 접속).

- 이미지 위에서 **주행 가능한 영역을 직접 클릭**하거나, 오른쪽 체크박스로 골라도 됨 (여러 개 선택 가능
  → 합집합이 최종 마스크가 됨).
- "이 프레임엔 주행 가능 영역 없음" 체크 후 저장하면 빈 마스크로 저장됨.
- **저장** 누르면 자동으로 다음 미완료 프레임으로 이동. **건너뛰기**는 저장 없이 다음으로만 이동.
- 다 하고 나면 터미널에서 `Ctrl+C`로 서버 종료.

같은 명령을 다른 번호로 계속 반복하면 됨: `./labeling.sh 0036`, `./labeling.sh 0037` ...
번호별로 새로 시작하는 게 아니라 **기존 라벨 위에 계속 쌓인다** (아래 저장 위치 참고).

## 2) 라벨링 결과 저장 위치

전부 아래 폴더 하나에 누적된다 (특정 데이터셋 번호로 나뉘지 않음 — sample_id가
`<ride_id>_f<frame_idx>` 형식이라 여러 tar를 걸쳐 라벨링해도 안 섞이고 계속 쌓임):

```
scripts/GeNIE_ws/pre_processing_dataset_withSAM2/data/
├── frames/<sample_id>.jpg       # 원본 프레임 (작업 해상도로 리사이즈)
├── proposals/<sample_id>.npz    # SAM2가 만든 mask 후보들(01번 결과)
├── labels/<sample_id>.png       # 사람이 고른 최종 traversability mask (0=불가, 255=가능) — 실제 "라벨"
└── manifest.csv                 # sample_id, ride, frame_idx, w, h, num_proposals, annotated(0/1)
```

- `annotated=1`인 행만 "라벨링 완료"된 것으로 취급되고, 학습(`training.sh`)도 이 조건으로 걸러서 씀.
- 이 폴더는 용량 때문에 git에 커밋 안 함(`.gitignore`에 이미 포함).

## 3) 학습: `training.sh`

```bash
./training.sh          # 기본 60 epoch
./training.sh 100      # epoch 수 직접 지정
```

특정 데이터셋 번호만 골라 학습하는 옵션은 없고, **그동안 쌓인 라벨 전체**(`manifest.csv`의
`annotated=1` 전부)를 대상으로 학습한다 — `03_train_sam_tp.py`가 원래 그렇게 동작하도록 되어 있어서
따로 손댈 필요 없었음.

### 결과물

```
runs/sam_tp/best_sam_tp.pt    # val loss 기준 최고 체크포인트
runs/sam_tp/loss_curve.png    # train/val loss·IoU 곡선 (과적합 여부 확인용, 꼭 볼 것)
runs/sam_tp/eval/*.png        # 검증셋 [원본 | 정답 | 예측] 비교 이미지
```

## 4) 디스크 용량 관리

`minirover-<ID>.tar`는 개당 약 2GB. 압축 해제까지 정상적으로 끝난 게 확인되면(`data/minirover-<ID>/`
안에 ride 폴더들이 다 보이면) tar 파일은 지워도 된다 — 데이터는 이미 `data/`에 풀려있으므로:

```bash
rm minirover-0011.tar
```

(단, 이 tar들은 팀 내부 공유 파일이라 재다운로드 URL이 없음 — 지우기 전에 압축 해제가 완전히
끝났는지 확인하고 지울 것. `df -h /`로 디스크 여유공간 수시로 확인 권장.)

tar 파일은 `.gitignore`에 `/minirover-*.tar` 패턴으로 등록되어 있어 어떤 번호든 실수로 git에
커밋되지 않는다.
