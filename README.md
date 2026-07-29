# frodobots-earth-rover

FrodoBots 소형 로버(Earth Rover) 주행 데이터를 가지고, [GeNIE](https://clear-nus.github.io/genie/)
(Wang et al., "GeNIE: A Generalizable Navigation System for In-the-Wild Environments",
IEEE RA-L 2025 — ICRA 2025 Earth Rover Challenge 1위) 논문의 파이프라인
(traversability 예측 -> BEV projection -> 후보 경로 샘플링/path fusion)을
재현하고, BEV 위에서 실제 주행 행동을 학습해보는 실험 저장소. 여기에 더해
`scripts/GeNIE_ws/pre_processing_dataset_withSAM2/`에서는 논문 Sec III-B의
반자동 라벨링 파이프라인(SAM2 후보 생성 -> 사람이 웹 UI로 선택 -> 파인튜닝)을
그대로 재현해서, 우리 데이터로 직접 "진짜" SAM-TP를 만들어본다.

참고 논문 원문(`GeNIE.pdf`)과 데이터셋 tar(`minirover-0011.tar`)는 저장소 루트에
두고 로컬에서만 쓴다 — 둘 다 용량/저작권 문제로 git에는 커밋하지 않는다
(`.gitignore` 참고).

## 데이터셋

### 1) frodobots-dataset-getting-started (공식 샘플, 세션 2개)

```bash
bash scripts/download_data.sh
```
`data/frodobots-dataset-getting-started/` 아래에 자동으로 받아서 압축까지 풀어준다.

### 2) minirover-0011 (메인 데이터셋, ride 15개)

`minirover-0011.tar`(팀 내부 공유 파일 — 별도 다운로드 URL 없음)는 저장소 루트,
즉 `./minirover-0011.tar` 경로에 그대로 두면 된다. 이 저장소는 이미 전체 15개
ride를 `data/minirover-0011/`에 압축 해제까지 완료해둔 상태다 (비압축 총 용량 약
1.9GB). tar 파일 자체와 압축 해제된 `data/minirover-0011/`은 용량 때문에 git에는
커밋하지 않으므로(`.gitignore` 참고), 새로 clone한 환경이거나 다시 풀어야 할
경우엔 아래 명령을 그대로 쓰면 된다:

```bash
# minirover-0011.tar 를 저장소 루트(이 README와 같은 위치)에 둔 다음:
mkdir -p data/minirover-0011
tar -xf minirover-0011.tar -C data/minirover-0011
```

압축을 풀면 `data/minirover-0011/ride_<id>_<code>_<timestamp>/` 형태로 ride별
폴더가 15개 생긴다. 각 ride 폴더 구성:

- `control_data_<id>.csv` — `linear, angular, rpm_1~4, timestamp`
- `gps_data_<id>.csv`(timestamp는 다른 csv와 달리 ms 단위), `imu_data_<id>.csv`
- `front_camera_timestamps_<id>.csv`, `rear_camera_timestamps_<id>.csv`
- `recordings/*.ts` — HLS 비디오 세그먼트. 파일명의 `uid_s_1000`=전면 카메라
  (1280x720), `uid_s_1001`=후면 카메라(640x360)

용량이 아깝다면 전체 대신 `tar -xf minirover-0011.tar -C data/minirover-0011
"ride_<원하는_ID>_*"` 로 특정 ride만 골라서 풀어도 된다 (`04`~`06`번 스크립트
모두 ride 폴더 하나 단위로 동작하므로 일부만 풀어도 문제없다).

## scripts/04_bev_traversability.py — traversability + BEV projection 디버그 도구

GeNIE 논문 Fig.3 파이프라인 중 **"SAM-TP traversability 예측 -> BEV cost map projection"**
구간까지 재현하고, 입력/추정/BEV를 한눈에 볼 수 있는 3분할 디버그 이미지를
`debug/ride_<id>/frame_<번호>_bev.png` 로 저장하는 스크립트. (그 다음 단계인 path
sampling/fusion은 이 스크립트 범위 밖 — `project_to_bev()`가 격자형 numpy 배열을
반환하므로 그 위에 바로 이어 붙일 수 있게만 해뒀다.)

### 개념

**1) Traversability score 추정** (`estimate_traversability` / `Sam2TraversabilityEstimator`)
픽셀마다 "이 지점이 로봇이 지나갈 수 있는 곳인가"를 0~1 확률로 계산한다.

- `--estimator sam2` (기본값): 원본 SAM2(`facebook/sam2.1-hiera-tiny`, 논문과 동일
  백본, **파인튜닝 없음**)를 그대로 불러와 쓴다. SAM2는 "traversable area" 같은
  추상 개념을 텍스트로 이해하지 못하고 point/box 프롬프트가 필요하므로, 논문이
  학습 라벨에 강제한 규칙("로봇은 항상 화면 하단-중앙 근처에 있다")을 그대로
  이용해 하단-중앙 근처에 foreground point 3개를 고정 프롬프트로 찍는다. SAM2가
  내놓는 후보 마스크 3개 중 예측 IoU가 가장 높은 것을 고르고, 그걸 이진화하지
  않고 로짓에 sigmoid만 씌워서 "확률" 그대로 사용한다. 포장도로·실내 바닥 같은
  구조화된 장면에서는 장애물/사람 다리까지 픽셀 단위로 정확히 구분하지만, 논문의
  SAM-TP처럼 traversability 데이터로 파인튜닝된 게 아니라서 잔디밭 같은 비정형
  지형은 "시각적으로 이어진 지면"이라는 이유만으로 과대평가하는 경향이 있다(한계).
- `--estimator heuristic`: torch/GPU 없이도 도는 대안. 하단-중앙 박스를 로봇 발밑
  시드로 잡고, Lab 색공간 거리 + Sobel 텍스처 그라디언트 + 지평선 억제를 조합한
  classical CV 휴리스틱. SAM2보다 정확도는 낮음.
- `--estimator sam_tp`: `scripts/GeNIE_ws/pre_processing_dataset_withSAM2/03_train_sam_tp.py`
  로 우리 라벨에 파인튜닝한 **진짜 SAM-TP** 체크포인트(`SamTPTraversabilityEstimator`)를
  쓴다. `sam2`와 달리 point prompt를 아예 안 주고, 파인튜닝된 학습 가능한 prompt
  token(`not_a_point_embed`/`no_mask_embed`)만으로 추론한다 — 자세한 원리는 아래
  섹션과 `03_train_sam_tp.py` docstring 참고.

**2) BEV(bird's-eye-view) projection** (`project_to_bev`)
depth 카메라가 없으므로, "카메라 높이 h + 로컬하게 평평한 지면"이라는 가정만으로
각 픽셀의 카메라 광선이 지면과 만나는 지점을 역산한다(논문 수식 `s·d_y = -h`와 동일
원리). 논문은 어안렌즈까지 보정하는 generic camera model을 쓰지만, 캘리브레이션
데이터가 없어 여기서는 표준 핀홀 모델 + 수평 화각(`--hfov`)으로 근사한다.

**3) BEV gap-filling** (`fill_bev_gaps`)
카메라 픽셀 해상도 한계로 먼 거리일수록 지면에 투영되는 픽셀이 희박해져 BEV 격자에
"정보 없음" 구멍이 많이 생긴다. 이를, (a) 카메라 시야각 안쪽이고 (b) 위/아래로 실측된
두 셀 사이(외삽 아님)이고 (c) 그 간격이 `--bev-fill-max-gap`(기본 1.5m) 이내인
경우에만 선형보간으로 채운다. 시야각 밖(카메라가 애초에 못 보는 영역)은 채우지
않으며, 보간된 셀은 시각화에서 옅은 톤으로 실측값과 구분해서 보여준다.

**4) 디버그 시각화** (`render_debug_panel`)
`[입력 이미지 | traversability 확률 히트맵 | BEV cost map]` 3분할 이미지 한 장으로
저장. BEV 패널에는 1m 간격 격자선, 로봇 위치(하단 중앙) 마커, `observed/interpolated/
unknown` 셀 개수를 헤더에 같이 적어준다.

### 실행 방법

`--estimator sam2`(기본값)는 `torch` + `transformers`(Sam2Model 포함 버전) + GPU가
필요하다. base conda env에는 없고 `isaaclab51` env에 이미 설치돼 있으므로 그
인터프리터로 실행하는 게 가장 빠르다:

```bash
# SAM2 기반 (기본값, 정확도 높음, GPU 사용)
/home/kante/miniconda3/envs/isaaclab51/bin/python3 scripts/04_bev_traversability.py \
    data/minirover-0011/ride_104913_dgp79m_20250228025658 \
    --num-samples 6 --cam-height 0.25 --hfov 110

# 휴리스틱 기반 (가볍게, GPU/torch 불필요 — numpy + opencv-python만 있으면 됨)
python3 scripts/04_bev_traversability.py \
    data/minirover-0011/ride_104913_dgp79m_20250228025658 \
    --estimator heuristic --num-samples 6
```

세션 경로 인자를 생략하면 `data/` 아래에서 `ride_*` 폴더를 자동으로 하나 찾아서
사용한다. 주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--estimator {sam2,heuristic,sam_tp}` | `sam2` | traversability 추정 방식 |
| `--sam-tp-checkpoint` | `runs/sam_tp/best_sam_tp.pt` | `--estimator sam_tp`일 때 쓸 체크포인트 경로 |
| `--num-samples` | `6` | 세션 전체에서 균등 샘플링할 프레임 수 |
| `--frame-indices` | - | 프레임 인덱스를 직접 지정 (지정 시 `--num-samples` 무시) |
| `--cam-height` | `0.25` (m) | 카메라 지면 높이 — 로버 실측값으로 바꾸는 걸 권장 |
| `--hfov` | `110` (deg) | 수평 화각 (핀홀 근사) |
| `--bev-range` | `8.0` (m) | BEV 전방/좌우 범위 (논문의 8m 플래닝 호라이즌 참고) |
| `--bev-resolution` | `0.05` (m) | BEV 셀 한 변 크기 |
| `--no-bev-fill` | off | gap-filling 끄고 순수 투영 결과만 보기 |
| `--out` | `debug/` | 결과 저장 루트 폴더 |

전체 옵션은 `--help` 로 확인. 결과는 `debug/ride_<id>/frame_<프레임번호>_bev.png`
에 저장된다.

## scripts/06_path_planning.py — 후보 경로 샘플링 + path fusion + 선택

`04`번이 만든 BEV cost map 위에서 GeNIE 논문 Sec III-D(Algorithm 1)의 경로 계획을
재현한다: 부채꼴로 후보 경로(2차 곡선) M개 샘플링 -> BEV 비용 기준 top-K 선별 ->
adaptive k-means(실루엣 점수로 k 자동 결정)로 클러스터링 -> 가까운 클러스터
병합(path fusion) -> 목표 방향과 heading이 가장 가까운 경로 선택. 결과를
`[입력 | traversability | BEV | BEV+경로]` 4분할 이미지로
`debug/ride_<id>/frame_<번호>_path.png` 에 저장한다 (후보=흰색, 선택된 경로=마젠타,
목표 방향=패널 가장자리의 초록 점).

**이 데이터셋에는 논문의 "GPS goal"이 없다** — 대회처럼 목표 체크포인트 좌표가
주어지는 게 아니라 그냥 주행 기록만 있다. 그래서 GPS 궤적에서 "최근 이동 방향"과
"몇 초 뒤 실제로 도달한 위치"를 뽑아 목표 방향의 대용으로 쓴다
(`estimate_goal_angle_deg`, `--goal-lookahead-s`로 몇 초 뒤를 볼지 조절, 또는
`--goal-angle-deg`로 직접 각도를 고정해서 이 추정을 건너뛸 수도 있음). 이런
설계상 차이점은 스크립트 최상단 docstring에 더 자세히 적어뒀다.

```bash
python3 scripts/06_path_planning.py \
    data/minirover-0011/ride_104913_dgp79m_20250228025658 --num-samples 6
```

`04`번과 동일한 BEV 옵션(`--cam-height`, `--hfov`, `--estimator` 등)에 더해 주요
경로 계획 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--n-candidates` | `25` | 샘플링할 후보 경로 수 |
| `--top-k` | `8` | BEV 비용 기준 상위 K개만 클러스터링에 사용 |
| `--k-max` | `6` | adaptive k-means가 시도할 최대 클러스터 수 |
| `--merge-threshold` | `0.6` (m) | 클러스터 centroid 병합 임계값(평균 waypoint 거리) |
| `--goal-lookahead-s` | `8.0` | GPS 목표 방향 추정 시 몇 초 뒤 위치를 볼지 |
| `--goal-angle-deg` | - | 목표 방향을 GPS 추정 대신 직접 고정값[deg]으로 지정 |

## scripts/05_train_bev_policy.py — BEV cost map만으로 조향 학습해보기

"BEV cost map 한 장만 보고 실제 사람이 조종한 (linear, angular)를 얼마나 맞출 수
있는가"를 검증하는 간단한 behavior-cloning 실험. `04`번의 traversability/BEV
projection 로직을 그대로 재사용해 여러 ride의 프레임마다 (BEV cost map, 실측
조종값) 쌍을 모으고(`.npz`로 캐싱), 작은 CNN으로 회귀 학습한다. 학습 후 검증셋
일부를 골라 BEV 위에 정답(초록 화살표)/예측(마젠타 화살표) 방향을 같이 그려
`debug/bev_policy_eval/`에 저장한다 — "확률 높은 쪽으로 가는지"를 눈으로 확인하기
위함.

```bash
# heuristic 추정기로 데이터셋 생성 + 학습 (빠름)
python3 scripts/05_train_bev_policy.py --epochs 20

# SAM2 추정기로 (정확하지만 느림 -> stride를 크게)
/home/kante/miniconda3/envs/isaaclab51/bin/python3 scripts/05_train_bev_policy.py \
    --estimator sam2 --frame-stride 60 --max-frames-per-ride 80 --epochs 20
```

결과물: `runs/bev_policy/best_bev_policy.pt`(체크포인트), `loss_curve.png` /
`val_predictions.png`(학습 곡선·정답-예측 산점도), `debug/bev_policy_eval/`(화살표
시각화). BEV 데이터셋 캐시는 기본 `bev_policy_data/dataset.npz` — 둘 다 용량이
커질 수 있어 git에는 커밋하지 않는다.

## scripts/GeNIE_ws/pre_processing_dataset_withSAM2/ — SAM-TP 라벨링 + 파인튜닝

GeNIE 논문 Sec III-B의 반자동 라벨링 파이프라인을 그대로 재현한다:

```
원본 이미지 -> SAM2로 여러 영역 mask 자동 생성 -> 사람이 주행 가능한 영역을 선택 -> 최종 traversability mask -> SAM2 파인튜닝(SAM-TP)
```

**1) `01_generate_mask_proposals.py`** — `data/` 안 ride들에서 프레임을 골라(기본
50장 목표, ride별로 고르게 분배) SAM2에 7x6 격자 point prompt를 던져 프레임당
10~20개의 서로 다른 영역 후보를 자동 생성한다(이미지 인코더는 프레임당 1번만,
디코더만 배치로 여러 번 돌려서 빠름). 결과는 `data/frames/`, `data/proposals/`,
`data/manifest.csv`에 저장.

```bash
python3 scripts/GeNIE_ws/pre_processing_dataset_withSAM2/01_generate_mask_proposals.py --num-frames 50
```

**2) `02_annotate_app.py`** — 로컬 Flask 웹앱(`http://localhost:5050`)에서 후보
영역들의 경계선(번호 포함)을 보여주고, **이미지를 직접 클릭**하거나 오른쪽
체크박스로 "주행 가능한 영역"을 고르면 초록으로 실시간 하이라이트된다(여러 개
선택 -> 합집합). 저장하면 `data/labels/<sample_id>.png`(최종 마스크)로 저장되고
자동으로 다음 미완료 프레임으로 이동한다.

```bash
python3 scripts/GeNIE_ws/pre_processing_dataset_withSAM2/02_annotate_app.py
# 원격 서버라면: ssh -L 5050:localhost:5050 <host> 로 포트포워딩 후 로컬 브라우저 접속
```

**3) `03_train_sam_tp.py`** — 라벨링된 이미지로 SAM2를 실제 "SAM-TP"로 파인튜닝.
핵심 트릭: point/box를 하나도 안 주고 `Sam2Model`을 호출하면, HuggingFace
transformers 구현이 자동으로 학습 가능한 `not_a_point_embed`/`no_mask_embed`
파라미터를 prompt로 쓴다 — 이게 논문이 말하는 "traversable 개념을 담은 학습
가능한 prompt token"과 사실상 동일한 메커니즘이라, Sam2Model을 수정하지 않고도
표준 파인튜닝 루프로 SAM-TP를 재현할 수 있다. 데이터가 (논문의 15,347장과 달리)
수십 장 수준으로 매우 적으므로, 기본값은 **이미지 인코더(백본)를 동결**하고
prompt token + mask_decoder만 학습한다(`--unfreeze-encoder`로 바꿀 수 있으나
권장 안 함). val loss 기준 최고 체크포인트를 자동 저장하며(과적합 이전 시점을
잡기 위함), 학습 후 val 세트에 대한 [원본 | 정답 | 예측] 정성 비교 이미지도 같이
저장한다.

```bash
python3 scripts/GeNIE_ws/pre_processing_dataset_withSAM2/03_train_sam_tp.py --epochs 60
```

결과물: `runs/sam_tp/best_sam_tp.pt`(체크포인트), `loss_curve.png`(loss/IoU
곡선), `eval/<sample_id>.png`(정성 비교). 이 체크포인트는 바로 `04`/`06`번
스크립트에서 `--estimator sam_tp`로 불러와 실제 BEV 파이프라인에 넣어볼 수 있다.

⚠️ 라벨이 수십 장 수준이면 train loss/IoU는 계속 좋아지는데 val은 금방
나빠지는(과적합) 게 정상이다 — `loss_curve.png`로 꼭 확인할 것. 더 정확한
모델을 원하면 01번으로 라벨을 더 만들고(특히 "주행 불가" 사례, 다양한 지형) 02번
으로 라벨링을 늘린 뒤 재학습하면 된다.

## 디렉토리 구조 / git에 안 올리는 것들

`.gitignore`로 다음을 제외한다 — 전부 로컬에서 재생성 가능하거나(스크립트 재실행),
용량이 크거나(데이터셋), 저작권이 걸린(논문 PDF) 것들이라 git 이력에 넣지 않는다:

- `GeNIE.pdf`, `minirover-0011.tar` — 원본 자료, 저장소 루트에 로컬로만 보관
- `data/` — 압축 해제된 데이터셋
- `debug/`, `runs/`(`bev_policy/`, `sam_tp/` 포함), `bev_policy_data/` — 스크립트 실행 결과물
- `scripts/GeNIE_ws/pre_processing_dataset_withSAM2/data/` — 01/02번이 만드는
  프레임/SAM2 후보/라벨 (코드는 커밋하되 이미지 데이터는 제외)
- `*.pt`, `*.npz` — 체크포인트/캐시 파일
- `__pycache__/`, `*.pyc`
