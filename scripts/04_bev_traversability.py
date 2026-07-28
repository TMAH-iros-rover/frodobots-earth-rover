"""
04_bev_traversability.py
GeNIE (Wang et al., RA-L 2025) 논문의 파이프라인을 참고하여,
FrodoBots 세션의 전면 카메라 프레임 -> (1) traversability score map
-> (2) BEV(bird's-eye-view) cost map 투영 까지 재현하고,
결과를 디버그용 3분할 이미지로 debug/ 폴더에 저장하는 스크립트.

[논문과의 대응 관계 요약 (Fig. 3, Sec III-B/III-C)]
  1. SAM-TP 모듈: RGB 이미지 -> pixel-wise traversability score.
     논문은 SAM2를 traversability 데이터셋(15,347장)으로 파인튜닝한 SAM-TP를 쓰지만,
     그 가중치는 이 저장소에 없다. 대신 --estimator sam2 (기본값)를 쓰면 파인튜닝
     안 된 원본 SAM2(facebook/sam2.1-hiera-tiny, 논문과 동일 백본)를 그대로 불러와,
     논문이 학습 라벨에 강제한 제약("로봇이 지금 서 있는 바닥-중앙 지점에서 곧장
     이어지는 영역만 traversable")을 point prompt로 흉내낸다: 화면 하단-중앙 근처에
     foreground point 3개를 찍어 SAM2에 넣고, 후보 마스크 3개 중 IoU 예측 점수가
     가장 높은 것을 고른 뒤 (이진화하지 않고) 로짓에 sigmoid를 씌운 채 그대로
     반환한다 -> 결과는 "그 픽셀이 traversable일 확률"이라는 진짜 확률 맵이 된다.
     (SAM2가 텍스트 프롬프트를 지원하지 않아 point/box가 필요하다는 점, 논문이
     VLM으로 점을 구하던 것을 여기서는 고정 휴리스틱 점으로 대체했다는 점은
     Sam2TraversabilityEstimator 문서에 자세히 적어뒀다.)
     --estimator heuristic 을 쓰면 GPU/torch 없이도 도는 color/texture 기반
     휴리스틱(estimate_traversability)으로 대체할 수 있다 (정확도는 SAM2보다 낮음).
     -> 실제 SAM-TP 파인튜닝 체크포인트가 생기면, 두 추정기와 동일한 시그니처
        (이미지 -> HxW float32 [0,1] 맵)로 새 추정기를 추가하면 바로 교체된다.
  2. BEV Projection (Sec III-C, 수식 s*d_y = -h):
     depth 카메라가 없으므로, "카메라 높이 h + 로컬 평면 지면 가정"만으로
     각 픽셀의 광선(ray) 방향 벡터가 지면과 만나는 3D 지점을 역산한다.
     논문은 어안렌즈 왜곡까지 보정하는 generic camera model[29]을 쓰지만,
     여기서는 캘리브레이션 데이터가 없으므로 표준 핀홀 모델 + 수평 FOV 파라미터로
     근사한다 (--hfov 로 조정 가능).
  3. Path sampling/fusion(Sec III-D)은 이번 스크립트의 범위 밖이다(요청 범위:
     BEV 투영 + traversability score 까지). BEV cost map만 있으면 그 위에서
     바로 이어붙일 수 있도록 project_to_bev()가 격자(grid) 형태의 numpy 배열을 반환한다.

사용 예:
  # SAM2 기반 (기본값, torch+transformers+GPU 필요 -> isaaclab51 conda env 사용 권장)
  /home/kante/miniconda3/envs/isaaclab51/bin/python3 scripts/04_bev_traversability.py \
      data/minirover-0011/ride_104913_dgp79m_20250228025658 --num-samples 6

  # 휴리스틱 기반 (torch 없이, numpy+opencv만으로 가볍게)
  python3 scripts/04_bev_traversability.py <ride_dir> --estimator heuristic

  인자 없이 실행하면 data/ 아래에서 ride_* 세션을 자동으로 찾아 사용한다.

필요 패키지:
  - heuristic 추정기: numpy, opencv-python(-headless)
  - sam2 추정기(기본): 위 두 개 + torch, transformers>=4.5 (Sam2Model 포함 버전)
"""
import os
import re
import sys
import glob
import argparse
import csv

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
DEBUG_DIR = os.path.join(REPO_ROOT, "debug")


# --------------------------------------------------------------------------
# 1) 세션 탐색 / 프레임 로딩
#    - FrodoBots 데이터는 두 가지 형태가 섞여 있다:
#      (a) front_camera_<rid>.mp4 하나로 합쳐진 형태 (frodobots-dataset-getting-started)
#      (b) recordings/ 아래 HLS .ts 세그먼트로 쪼개진 형태 (minirover-0011.tar)
#    아래 함수들은 두 형태를 모두 "순서가 있는 비디오 소스 리스트"로 통일해서 다룬다.
# --------------------------------------------------------------------------

def find_ride_dirs(root=DATA_DIR):
    """data/ 아래에서 ride_* 로 시작하는 세션 폴더를 재귀적으로 모두 찾는다.
    (인자 없이 스크립트를 실행했을 때 자동으로 세션을 고르기 위한 용도)"""
    cand = sorted(glob.glob(os.path.join(root, "**", "ride_*"), recursive=True))
    return [c for c in cand if os.path.isdir(c)]


def parse_ride_id(session_dir):
    """폴더명 'ride_<id>_...' 에서 숫자 id만 뽑아낸다.
    getting-started 형식(ride_19154_20240225023555)과
    minirover 형식(ride_104913_dgp79m_20250228025658) 둘 다 index=1이 id다."""
    name = os.path.basename(session_dir.rstrip("/"))
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"세션 폴더명에서 ride id를 못 찾음: {name}")
    return parts[1]


def _ts_from_ts_filename(path):
    """'..._video_20250228025659893.ts' 꼴 파일명 뒤쪽의 숫자(수집 시각)를 정렬 키로 추출."""
    m = re.search(r"(\d+)\.ts$", os.path.basename(path))
    return int(m.group(1)) if m else 0


def find_front_camera_sources(session_dir, rid):
    """전면 카메라 비디오 소스를 (경로 리스트, 설명) 형태로 반환.
    우선순위: (1) 단일 mp4  (2) uid_s_1000 비디오 세그먼트(.ts, minirover 데이터셋에서
    실측 확인한 전면 카메라 uid)  (3) 그래도 없으면 오디오가 아닌 video 세그먼트들을
    uid별로 묶어 평균 파일 크기가 가장 큰 그룹을 전면 카메라로 추정(고해상도일수록
    전면 카메라일 가능성이 높다는 휴리스틱)."""
    mp4 = os.path.join(session_dir, f"front_camera_{rid}.mp4")
    if os.path.isfile(mp4):
        return [mp4], "front_camera mp4"

    rec_dir = os.path.join(session_dir, "recordings")
    ts_files = sorted(glob.glob(os.path.join(rec_dir, "*uid_s_1000*video*.ts")),
                       key=_ts_from_ts_filename)
    if ts_files:
        return ts_files, "uid_s_1000 HLS segments"

    all_video_ts = glob.glob(os.path.join(rec_dir, "*video*.ts"))
    groups = {}
    for f in all_video_ts:
        m = re.search(r"uid_s_(\d+)__uid_e_video", os.path.basename(f))
        if m:
            groups.setdefault(m.group(1), []).append(f)
    if not groups:
        raise FileNotFoundError(f"전면 카메라 소스를 찾을 수 없음: {session_dir}")
    best_uid = max(groups, key=lambda u: sum(os.path.getsize(f) for f in groups[u]))
    ts_files = sorted(groups[best_uid], key=_ts_from_ts_filename)
    return ts_files, f"uid_s_{best_uid} HLS segments (크기 기반 추정)"


def count_frames_hint(session_dir, rid):
    """전체 프레임 수 추정치. front_camera_timestamps_<rid>.csv 행 수를 우선 사용
    (실제 비디오를 끝까지 디코딩하지 않아도 되므로 훨씬 빠르다)."""
    ts_csv = os.path.join(session_dir, f"front_camera_timestamps_{rid}.csv")
    if os.path.isfile(ts_csv):
        with open(ts_csv, newline="") as f:
            n = sum(1 for _ in csv.reader(f)) - 1  # 헤더 제외
        if n > 0:
            return n
    return None


def extract_frames_at_indices(sources, indices):
    """여러 비디오 소스(mp4 1개 또는 .ts 세그먼트 여러 개)를 이어붙인 것으로 보고,
    전역 프레임 인덱스 기준으로 원하는 프레임들만 뽑아 dict{index: frame}으로 반환.
    HLS .ts는 임의 seek가 불안정하므로 순차 디코딩하되, 필요한 마지막 인덱스를
    지나면 즉시 중단해 불필요한 디코딩을 줄인다."""
    wanted = set(int(i) for i in indices)
    if not wanted:
        return {}
    max_idx = max(wanted)

    out = {}
    global_idx = 0
    for src in sources:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            continue
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if global_idx in wanted:
                out[global_idx] = frame
            global_idx += 1
            if global_idx > max_idx:
                break
        cap.release()
        if global_idx > max_idx:
            break
    return out


# --------------------------------------------------------------------------
# 2) Traversability score 추정 (SAM-TP의 오프라인 자리표시자)
# --------------------------------------------------------------------------

def estimate_traversability(image_bgr, horizon_ratio=0.45, seed_box_frac=(0.35, 0.65, 0.80, 0.98),
                             color_sigma=2.2, texture_weight=0.35, blur_frac=0.008):
    """이미지 한 장 -> 픽셀별 traversability score(HxW, float32, 0~1, 1=이동 가능).

    로직 (GeNIE 논문의 데이터셋 설계 제약을 휴리스틱으로 재현):
      1) 이미지 하단-중앙 박스를 "로봇이 지금 서 있는 바닥"의 시드(seed) 영역으로 가정한다.
         (논문 Sec III-B: "traversable mask includes only regions that are directly
         navigable from the robot's current position, typically located near the
         bottom center of the image" — 즉 학습 라벨 자체가 이렇게 만들어졌다는 점을
         그대로 흉내낸다.)
      2) 시드 영역의 Lab 색공간 평균/표준편차를 구하고, 이미지 전체 픽셀이 그 색
         분포에서 얼마나 벗어나는지를 정규화 거리로 계산해 색상 유사도 점수로 쓴다.
         (같은 재질의 바닥/도로는 색이 비슷하게 유지된다는 가정)
      3) Sobel 그라디언트 크기로 텍스처 급변(경계, 잡초, 자갈 등)을 검출해 점수를
         약간 깎는다 — 텍스처가 급격히 변하는 곳은 실제 지면이 아닐 가능성이 높다.
      4) 화면 상단(하늘/먼 배경)은 지면일 수 없으므로 --horizon-ratio 위쪽을
         부드러운 램프(선형 감쇠)로 0에 가깝게 눌러준다 (하드 컷 대신 그라데이션을
         써서 경계 아티팩트를 줄임).
      5) 마지막으로 가우시안 블러로 스무딩해 SAM2가 흔히 겪는 "조각난 마스크"
         문제(논문 Sec III-B) 대신 하나로 이어진 넓은 영역을 얻는다.

    이 함수의 입출력 시그니처(BGR 이미지 -> HxW [0,1] float32)만 지키면,
    나중에 실제 학습된 SAM-TP 추론 함수로 그대로 바꿔 끼울 수 있다.
    """
    h, w = image_bgr.shape[:2]
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    x0f, x1f, y0f, y1f = seed_box_frac
    sx0, sx1 = int(w * x0f), int(w * x1f)
    sy0, sy1 = int(h * y0f), int(h * y1f)
    seed = lab[sy0:sy1, sx0:sx1].reshape(-1, 3)
    if seed.size == 0:
        seed = lab.reshape(-1, 3)
    mean = seed.mean(axis=0)
    std = seed.std(axis=0) + 1e-3

    # 1) 색상 유사도: 시드 색 분포로부터의 정규화 거리 -> 가우시안 스코어
    diff = (lab - mean) / std
    dist2 = np.sum(diff * diff, axis=-1)
    color_score = np.exp(-dist2 / (2.0 * color_sigma ** 2))

    # 2) 텍스처(경계) 패널티
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad_norm = grad / (np.percentile(grad, 95) + 1e-6)
    grad_norm = np.clip(grad_norm, 0.0, 1.0)

    score = color_score * (1.0 - texture_weight * grad_norm)

    # 3) 지평선 위쪽(하늘/먼 배경) 억제: horizon_row 위는 0, 그 아래로 10%h 구간에서
    #    선형으로 1까지 램프업
    horizon_row = int(h * horizon_ratio)
    ramp_band = max(1, int(h * 0.10))
    rows = np.arange(h, dtype=np.float32)
    vertical_prior = np.clip((rows - horizon_row) / ramp_band, 0.0, 1.0)
    score *= vertical_prior[:, None]

    # 4) 스무딩 (조각난 마스크 대신 하나로 이어진 영역)
    sigma = max(1.0, w * blur_frac)
    score = cv2.GaussianBlur(score, (0, 0), sigmaX=sigma)

    return np.clip(score, 0.0, 1.0).astype(np.float32)


class Sam2TraversabilityEstimator:
    """원본 SAM2(facebook/sam2.1-hiera-tiny, 논문과 동일 백본)를 이용해 traversability
    "확률" 맵을 얻는 추정기. estimate_traversability()와 똑같이 (BGR 이미지) -> (HxW
    float32, [0,1]) 를 반환하므로 서로 바꿔 끼울 수 있다.

    로직:
      1) SAM2는 "traversable area"라는 추상 개념을 텍스트로 이해하지 못하고 point/box
         프롬프트가 반드시 있어야 마스크를 만든다(논문 Sec III-B가 지적한 한계 그대로).
         논문은 이 점을 보완하려고 VLM에게 먼저 "어디를 클릭해야 하는가"를 물어본 뒤
         그 점을 SAM2에 넣는다. 여기서는 VLM 대신, 논문이 학습 라벨 자체에 강제했던
         제약("로봇은 항상 화면 하단-중앙 근처에 있다")을 그대로 이용해 하단-중앙
         근처에 foreground point 3개(중앙 1개 + 좌우 2개)를 고정 프롬프트로 사용한다.
      2) SAM2는 한 프롬프트에 대해 보통 서로 다른 크기/모양의 후보 마스크 3개와 각각의
         예측 IoU 점수를 함께 출력한다(모호성 처리를 위한 SAM 계열 공통 설계). 그중
         예측 IoU가 가장 높은 마스크 1개를 선택한다.
      3) 선택된 마스크는 원래 최종 이진 마스크로 쓰라고 나온 256x256 해상도의 로짓
         (logit)이다. 이걸 곧바로 0/1로 이진화하지 않고 sigmoid를 씌워 "그 픽셀이
         traversable일 확률"로 남겨둔 채, 원본 해상도로 bilinear 업샘플링만 해서
         반환한다 -> 파이프라인의 나머지 단계(BEV projection 등)는 이 확률을 그대로
         cost로 사용한다.

    필요 패키지: torch, transformers (Sam2Model/Sam2Processor 포함 버전).
    GPU가 있으면 자동으로 사용한다 (없으면 CPU로 느리게 동작).
    """

    def __init__(self, model_id="facebook/sam2.1-hiera-tiny", device=None):
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor
        except ImportError as e:
            raise ImportError(
                "--estimator sam2 를 쓰려면 torch + transformers(Sam2Model 포함 버전)가 "
                "필요합니다. 예) /home/kante/miniconda3/envs/isaaclab51/bin/python3 로 "
                f"실행하거나 pip install torch transformers 로 설치하세요. (원인: {e})"
            ) from e
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[sam2] {model_id} 로딩 중 (device={self.device}) ...")
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(self.device).eval()
        print("[sam2] 로딩 완료")

    def __call__(self, image_bgr, seed_box_frac=(0.35, 0.65, 0.80, 0.98)):
        torch = self._torch
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        x0f, x1f, y0f, y1f = seed_box_frac
        points = [
            [int(w * (x0f + x1f) / 2), int(h * (y0f + y1f) / 2)],  # 중앙
            [int(w * x0f), int(h * y1f)],                           # 좌
            [int(w * x1f), int(h * y1f)],                           # 우
        ]
        labels = [1, 1, 1]  # 전부 foreground(=여기가 traversable) 점

        inputs = self.processor(images=image_rgb, input_points=[[points]], input_labels=[[labels]],
                                 return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, multimask_output=True)

        logits = out.pred_masks[0, 0]          # (num_candidate_masks, h_low, w_low)
        iou_pred = out.iou_scores[0, 0]        # (num_candidate_masks,)
        best = int(iou_pred.argmax().item())
        prob_low = torch.sigmoid(logits[best])
        prob = torch.nn.functional.interpolate(
            prob_low[None, None], size=(h, w), mode="bilinear", align_corners=False
        )[0, 0]
        return prob.detach().float().cpu().numpy()


def build_traversability_estimator(args):
    """--estimator 값에 따라 (BGR 이미지 -> HxW [0,1] 확률맵) 콜러블을 만들어 반환.
    SAM2 모델은 로딩 비용이 있으므로 프레임마다가 아니라 실행당 한 번만 만든다."""
    if args.estimator == "heuristic":
        return lambda img: estimate_traversability(img, horizon_ratio=args.horizon_ratio)
    if args.estimator == "sam2":
        return Sam2TraversabilityEstimator(model_id=args.sam2_model)
    raise ValueError(f"알 수 없는 --estimator: {args.estimator}")


# --------------------------------------------------------------------------
# 3) BEV(bird's-eye-view) projection
# --------------------------------------------------------------------------

def project_to_bev(trav_map, cam_height_m=0.25, hfov_deg=110.0, cam_pitch_deg=0.0,
                    bev_range_m=8.0, bev_resolution_m=0.05):
    """픽셀별 traversability map -> 지면(ground plane) BEV cost map으로 투영.

    논문 Sec III-C 의 핵심 아이디어를 그대로 구현:
      - depth 센서가 없으므로 "카메라 높이 h가 알려져 있다 + 바닥은 로컬하게
        평평한 평면이다"라는 두 가지 가정만으로 3D를 복원한다.
      - 각 픽셀에 대응하는 카메라 좌표계 광선 방향(단위벡터) d_hat를 구한 뒤,
        d_hat의 아래방향(y) 성분이 카메라 높이 h만큼 내려가는 지점에서 지면과
        만난다고 보고 스케일 s를 푼다:  s * d_hat_y = h  (카메라 좌표계: x=우측,
        y=아래, z=전방이 + 인 convention을 사용하므로 논문 식 s*d_y=-h 와 부호만
        다르고 동일한 관계식이다).
      - 논문은 어안렌즈까지 보정하는 generic camera model[29]로 d_hat를 구하지만,
        여기서는 캘리브레이션 파일이 없으므로 수평 FOV(--hfov)만으로 광선 방향을
        근사하는 표준 핀홀 모델을 쓴다. (핀홀 근사이므로 화면 가장자리 왜곡이 큰
        렌즈일수록 오차가 커진다는 점을 코드 사용자가 인지하고 있어야 한다.)
      - cam_pitch_deg 로 카메라가 수평선 대비 위/아래로 얼마나 기울어져 있는지도
        보정할 수 있게 했다(x축 기준 회전).

    반환:
      bev_score: (grid_h, grid_w) float32, 셀에 떨어진 traversability 평균값.
                 grid_h는 전방(0~bev_range_m), grid_w는 좌우(-range/2~+range/2).
                 배열의 row=0 이 가장 먼 전방, row=grid_h-1 이 로봇 바로 앞(가까움)
                 이 되도록 정렬해 두었다 (이미지처럼 위=먼 곳, 아래=로봇 위치).
      known_mask: (grid_h, grid_w) bool, 해당 셀에 실제로 투영된 픽셀이 있었는지
                 여부 (없으면 "정보 없음"이지 "이동 불가"가 아니므로 시각화에서
                 구분해야 한다).
      meta: 격자 크기/해상도 등 부가 정보 dict.
    """
    h, w = trav_map.shape
    hfov = np.radians(hfov_deg)
    fx = (w / 2.0) / np.tan(hfov / 2.0)
    fy = fx  # 정사각 픽셀 가정 (수직 FOV는 종횡비로 자동 결정됨)
    cx, cy = w / 2.0, h / 2.0

    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx = (us - cx) / fx
    dy = (vs - cy) / fy
    dz = np.ones_like(dx)
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / norm, dy / norm, dz / norm

    # 카메라 상하 기울기(pitch) 보정: x축 기준 회전. pitch>0 이면 카메라가 더 아래를 본다.
    pitch = np.radians(cam_pitch_deg)
    dy_r = dy * np.cos(pitch) + dz * np.sin(pitch)
    dz_r = -dy * np.sin(pitch) + dz * np.cos(pitch)

    valid = dy_r > 1e-4  # 아래를 향하는 광선만 유한한 거리에서 지면과 만난다
    s = np.zeros_like(dy_r)
    s[valid] = cam_height_m / dy_r[valid]

    lateral = dx * s          # +면 오른쪽 (m)
    forward = dz_r * s        # 전방 거리 (m)

    in_range = valid & (forward > 0) & (forward <= bev_range_m) & (np.abs(lateral) <= bev_range_m / 2.0)

    grid_n = max(1, int(round(bev_range_m / bev_resolution_m)))
    grid_h = grid_w = grid_n

    fwd_v = forward[in_range]
    lat_v = lateral[in_range]
    score_v = trav_map[in_range]

    col = np.clip(((lat_v + bev_range_m / 2.0) / bev_resolution_m).astype(np.int32), 0, grid_w - 1)
    row = np.clip((grid_h - 1 - (fwd_v / bev_resolution_m).astype(np.int32)), 0, grid_h - 1)

    accum = np.zeros((grid_h, grid_w), dtype=np.float64)
    counts = np.zeros((grid_h, grid_w), dtype=np.int32)
    np.add.at(accum, (row, col), score_v)
    np.add.at(counts, (row, col), 1)

    known_mask = counts > 0
    bev_score = np.zeros((grid_h, grid_w), dtype=np.float32)
    bev_score[known_mask] = (accum[known_mask] / counts[known_mask]).astype(np.float32)

    meta = dict(grid_h=grid_h, grid_w=grid_w, range_m=bev_range_m, resolution_m=bev_resolution_m,
                cam_height_m=cam_height_m, hfov_deg=hfov_deg, cam_pitch_deg=cam_pitch_deg)
    return bev_score, known_mask, meta


def fill_bev_gaps(bev_score, known_mask, hfov_deg, bev_range_m, max_gap_m=1.5):
    """BEV 격자의 '정보 없음' 구멍을 조건부로 보간해서 메운다 (개선 사항).

    문제: 카메라 픽셀은 유한한데 원근 투영 특성상, 로봇 바로 앞(가까운 거리)은
    좁은 지면 범위에 픽셀이 촘촘히 몰리는 반면, 지평선에 가까운 픽셀 행 하나가
    지면에서는 몇 미터씩 건너뛰는 넓은 범위에 대응한다. 그 결과 먼 거리 구간은
    실제로 투영되는 픽셀이 거의 없어 known_mask가 듬성듬성해진다 (앞서 저장한
    디버그 이미지에서 2~8m 구간이 대부분 회색으로 비어 보이는 이유).

    해결: "카메라가 애초에 볼 수 없는 영역(= 시야각 밖)"과 "볼 수는 있지만
    해상도 한계로 샘플이 못 떨어진 영역"을 구분한다.
      1) 각 BEV 셀 중심의 방위각(azimuth)을 계산해, hfov 이내인 셀만
         '관측 가능 쐐기(fov wedge)'로 인정한다 — 이 쐐기 밖은 카메라가
         물리적으로 볼 수 없는 영역이므로 절대 채우지 않는다.
      2) 쐐기 안에서, 같은 좌우(column) 위치의 위/아래(먼 거리/가까운 거리) 양쪽에
         실측 셀이 있고 그 간격이 max_gap_m 이내면, 두 실측값 사이를 선형보간한다
         (도로/지면은 국소적으로 연속적이라는 가정). 간격이 너무 멀거나 한쪽에만
         실측값이 있으면(외삽이 되는 경우) 채우지 않고 그대로 회색으로 남긴다.

    반환: (filled_score, filled_mask, interpolated_mask)
      filled_mask = known_mask OR 보간으로 채운 셀. interpolated_mask는 그 중
      "실측이 아니라 보간으로 채운" 셀만 표시 (시각화에서 실측과 구분하기 위함).
    """
    grid_h, grid_w = bev_score.shape
    res = bev_range_m / grid_h
    max_gap_cells = max(1, int(round(max_gap_m / res)))

    rows = np.arange(grid_h)
    cols = np.arange(grid_w)
    forward = (grid_h - 1 - rows + 0.5) * res      # row -> 전방 거리[m]
    lateral = (cols + 0.5) * res - bev_range_m / 2.0  # col -> 좌우 거리[m]
    fwd_grid, lat_grid = np.meshgrid(forward, lateral, indexing="ij")
    azim_deg = np.degrees(np.arctan2(lat_grid, np.maximum(fwd_grid, 1e-6)))
    fov_mask = np.abs(azim_deg) <= (hfov_deg / 2.0)

    filled_score = bev_score.copy()
    filled_mask = known_mask.copy()
    interpolated_mask = np.zeros_like(known_mask)

    for c in range(grid_w):
        known_rows_c = rows[known_mask[:, c]]
        if known_rows_c.size < 2:
            continue
        known_vals_c = bev_score[known_rows_c, c]
        lo, hi = known_rows_c.min(), known_rows_c.max()
        # 실측 셀 사이(외삽 아님) + 관측 가능 쐐기 안쪽만 채움 후보
        target_rows = rows[(~known_mask[:, c]) & fov_mask[:, c] & (rows > lo) & (rows < hi)]
        if target_rows.size == 0:
            continue
        idx_hi = np.searchsorted(known_rows_c, target_rows)
        idx_lo = idx_hi - 1
        row_lo, row_hi = known_rows_c[idx_lo], known_rows_c[idx_hi]
        gap = row_hi - row_lo
        ok = gap <= max_gap_cells
        if not np.any(ok):
            continue
        w = (target_rows[ok] - row_lo[ok]) / gap[ok]
        vals = known_vals_c[idx_lo[ok]] * (1 - w) + known_vals_c[idx_hi[ok]] * w
        tr = target_rows[ok]
        filled_score[tr, c] = vals
        filled_mask[tr, c] = True
        interpolated_mask[tr, c] = True

    return filled_score, filled_mask, interpolated_mask


# --------------------------------------------------------------------------
# 4) 디버그 시각화
# --------------------------------------------------------------------------

def _colorize(score01, unknown_mask=None, unknown_color=(40, 40, 40)):
    """0~1 스코어 맵을 jet 컬러맵(빨강=높은 traversability)으로 변환.
    논문 Fig.1의 'Red regions indicate high traversability score' 표현과 동일하게
    맞추기 위해 COLORMAP_JET을 사용한다."""
    u8 = np.clip(score01 * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    if unknown_mask is not None:
        color[unknown_mask] = unknown_color
    return color


def render_bev_panel(bev_score, known_mask, interpolated_mask, meta, panel_h=480, title=None):
    """BEV cost map을 "1m 간격 거리 눈금(왼쪽 여백에 tick만 표시) + 로봇 위치 마커 +
    전방축 점선"이 들어간 표준 패널 이미지로 렌더링한다. render_debug_panel()과
    06_path_planning.py가 이 함수를 공유해서 똑같은 스타일/좌표계를 쓰게 했다 —
    특히 06번 스크립트는 반환되는 geom(픽셀 좌표계 정보)을 이용해 후보/선택 경로를
    이 패널과 정확히 같은 위치에 겹쳐 그린다.

    거리 눈금은 실제 BEV 데이터 위에 선을 긋지 않고 왼쪽 바깥 여백에만 tick으로
    표시한다 (전폭 회색선을 그으면 '정보 없음' 회색과 헷갈리기 쉬워서 제거함).
    fill_bev_gaps()로 보간된 셀은 회색을 살짝 섞어 옅은 톤으로 표시해 실측값과
    구분한다.

    반환: (panel_img, geom). geom = dict(tick_margin, scale_px_per_m, origin_px,
    panel_h, range_m) — meter 좌표 (x=우측+, z=전방+) 를 이 패널의 픽셀 좌표로
    바꾸려면 `px = origin_px[0] + x*scale_px_per_m`, `py = origin_px[1] - z*scale_px_per_m`.
    """
    bev_color = _colorize(bev_score, unknown_mask=~known_mask)
    if interpolated_mask is not None and interpolated_mask.any():
        gray = np.array([150, 150, 150], dtype=np.float32)
        blended = bev_color[interpolated_mask].astype(np.float32) * 0.6 + gray * 0.4
        bev_color[interpolated_mask] = blended.astype(np.uint8)
    # 로봇 기준(BEV 하단 중앙)에서 전방이 위로 가도록 이미 row=0이 먼 곳이 되게 만들어뒀음.
    bev_up = cv2.resize(bev_color, (panel_h, panel_h), interpolation=cv2.INTER_NEAREST)
    grid_n = meta["grid_h"]
    res = meta["resolution_m"]
    scale = panel_h / grid_n

    tick_margin = 34
    bev_up = cv2.copyMakeBorder(bev_up, 0, 0, tick_margin, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    for d_m in range(0, int(meta["range_m"]) + 1):
        row_cell = grid_n - 1 - (d_m / res)
        y = int(row_cell * scale)
        if 0 <= y < panel_h:
            cv2.line(bev_up, (tick_margin - 6, y), (tick_margin, y), (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(bev_up, f"{d_m}m", (2, max(10, y + 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (255, 255, 255), 1, cv2.LINE_AA)
    # 로봇 전방축(x=0) 세로 점선 (왼쪽 눈금 여백만큼 cx를 오른쪽으로 보정)
    cx = tick_margin + panel_h // 2
    for y in range(0, panel_h, 8):
        cv2.line(bev_up, (cx, y), (cx, y + 4), (255, 255, 255), 1, cv2.LINE_AA)
    # 로봇 위치 마커 (하단 중앙 = 원점)
    origin_px = (cx, panel_h - 4)
    cv2.drawMarker(bev_up, origin_px, (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
    if title:
        cv2.putText(bev_up, title, (tick_margin + 10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)

    geom = dict(tick_margin=tick_margin, scale_px_per_m=panel_h / meta["range_m"],
                origin_px=origin_px, panel_h=panel_h, range_m=meta["range_m"])
    return bev_up, geom


def render_debug_panel(frame_bgr, trav_map, bev_score, known_mask, interpolated_mask, meta, header_lines,
                        heat_label="Traversability score", extra_panels=None):
    """[입력 이미지 | traversability heatmap | BEV cost map | (선택) extra_panels...]
    형태의 디버그 이미지를 생성한다. extra_panels에 (panel_h와 같은 높이의) BGR
    이미지 리스트를 넘기면 BEV 패널 오른쪽에 이어 붙인다 — 06_path_planning.py가
    "BEV + 경로" 패널을 여기 붙여서 한 이미지로 같이 저장하는 용도."""
    panel_h = 480

    def fit(img):
        s = panel_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), panel_h))

    p_input = fit(frame_bgr.copy())
    cv2.putText(p_input, "Input (front camera)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)

    heat = _colorize(trav_map)
    p_heat = fit(heat)
    cv2.putText(p_heat, heat_label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)

    res = meta["resolution_m"]
    bev_up, geom = render_bev_panel(
        bev_score, known_mask, interpolated_mask, meta, panel_h,
        title=f"BEV cost map (range={meta['range_m']:.0f}m, res={res*100:.0f}cm)",
    )
    cv2.putText(bev_up, "gray=no info, soft=interpolated", (geom["tick_margin"] + 10, panel_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    sep = np.full((panel_h, 6, 3), 60, dtype=np.uint8)
    panels = [p_input, sep, p_heat, sep, bev_up]
    for extra in (extra_panels or []):
        panels.append(sep)
        panels.append(extra)
    body = np.hstack(panels)

    header_h = 22 * (len(header_lines) + 1)
    header = np.zeros((header_h, body.shape[1], 3), dtype=np.uint8)
    for i, line in enumerate(header_lines):
        cv2.putText(header, line, (10, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([header, body])


# --------------------------------------------------------------------------
# 5) 파이프라인 실행
# --------------------------------------------------------------------------

def process_frame(frame_bgr, frame_idx, ride_id, out_dir, args, estimator):
    """프레임 한 장에 대해 traversability -> BEV projection -> 디버그 이미지 저장까지
    전체 파이프라인을 실행한다. estimator는 build_traversability_estimator()로 실행당
    한 번만 만들어서 여기로 전달한다 (SAM2 모델을 프레임마다 다시 로딩하지 않기 위함)."""
    trav_map = estimator(frame_bgr)
    bev_score, known_mask, meta = project_to_bev(
        trav_map, cam_height_m=args.cam_height, hfov_deg=args.hfov,
        cam_pitch_deg=args.cam_pitch_deg, bev_range_m=args.bev_range,
        bev_resolution_m=args.bev_resolution,
    )

    interpolated_mask = None
    if not args.no_bev_fill:
        bev_score, known_mask, interpolated_mask = fill_bev_gaps(
            bev_score, known_mask, hfov_deg=args.hfov, bev_range_m=args.bev_range,
            max_gap_m=args.bev_fill_max_gap,
        )

    n_known_before = int(known_mask.sum()) - (int(interpolated_mask.sum()) if interpolated_mask is not None else 0)
    n_filled = int(interpolated_mask.sum()) if interpolated_mask is not None else 0
    header = [
        f"ride={ride_id}  frame_idx={frame_idx}  estimator={args.estimator}",
        f"cam_height={args.cam_height:.2f}m  hfov={args.hfov:.0f}deg  pitch={args.cam_pitch_deg:.1f}deg  "
        f"bev_range={args.bev_range:.1f}m  bev_res={args.bev_resolution*100:.0f}cm",
        f"BEV cells: observed={n_known_before}  interpolated(+)={n_filled}  "
        f"unknown={known_mask.size - known_mask.sum()}  (total={known_mask.size})",
    ]
    heat_label = "Traversability prob. (SAM2)" if args.estimator == "sam2" else "Traversability score (heuristic)"
    panel = render_debug_panel(frame_bgr, trav_map, bev_score, known_mask, interpolated_mask, meta, header, heat_label)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}_bev.png")
    cv2.imwrite(out_path, panel)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", nargs="?", default=None,
                     help="ride_* 세션 폴더 경로 (생략 시 data/ 아래에서 자동 탐색)")
    ap.add_argument("--frame-indices", type=int, nargs="+", default=None,
                     help="처리할 프레임 인덱스를 직접 지정 (생략 시 --num-samples로 균등 샘플링)")
    ap.add_argument("--num-samples", type=int, default=6,
                     help="세션 전체에서 균등 간격으로 뽑을 프레임 개수 (기본 6)")
    ap.add_argument("--cam-height", type=float, default=0.25, help="카메라 지면 높이[m]")
    ap.add_argument("--hfov", type=float, default=110.0, help="수평 화각[deg] (핀홀 근사)")
    ap.add_argument("--cam-pitch-deg", type=float, default=0.0, help="카메라 아래쪽 기울기[deg], +면 아래를 봄")
    ap.add_argument("--bev-range", type=float, default=8.0, help="BEV 전방/좌우 범위[m] (논문 8m 플래닝 호라이즌 참고)")
    ap.add_argument("--bev-resolution", type=float, default=0.05, help="BEV 셀 한 변의 크기[m]")
    ap.add_argument("--horizon-ratio", type=float, default=0.45, help="이미지 상단 몇 %까지를 하늘/먼 배경으로 볼지 (0~1)")
    ap.add_argument("--bev-fill-max-gap", type=float, default=1.5,
                     help="BEV gap-filling 시 보간을 허용할 두 실측 셀 사이 최대 간격[m] (기본 1.5m)")
    ap.add_argument("--no-bev-fill", action="store_true",
                     help="BEV gap-filling(보간)을 끄고 순수 투영 결과만 보고 싶을 때 사용")
    ap.add_argument("--estimator", choices=["sam2", "heuristic"], default="sam2",
                     help="traversability 추정 방식: sam2(기본, torch+transformers+GPU 필요) "
                          "또는 heuristic(색상/텍스처 기반, 가벼움)")
    ap.add_argument("--sam2-model", default="facebook/sam2.1-hiera-tiny",
                     help="--estimator sam2 일 때 사용할 HuggingFace 모델 id (기본: 논문과 동일 백본)")
    ap.add_argument("--out", default=DEBUG_DIR, help="디버그 이미지 저장 루트 폴더 (기본: <repo>/debug)")
    args = ap.parse_args()

    if args.session_dir is None:
        candidates = find_ride_dirs()
        if not candidates:
            sys.exit("세션 경로를 인자로 넘겨줘: python 04_bev_traversability.py <ride_dir>")
        session_dir = candidates[0]
    else:
        session_dir = os.path.expanduser(args.session_dir)

    rid = parse_ride_id(session_dir)
    sources, src_desc = find_front_camera_sources(session_dir, rid)
    print(f"[session] {session_dir}")
    print(f"[ride_id] {rid}")
    print(f"[front_camera] {src_desc} ({len(sources)}개 소스 파일)")

    if args.frame_indices:
        indices = sorted(set(args.frame_indices))
    else:
        total = count_frames_hint(session_dir, rid)
        if total is None:
            total = 2000  # timestamps csv가 없을 때의 안전한 기본 상한
            print("[warn] front_camera_timestamps csv를 못 찾아 총 프레임수를 추정할 수 없음. "
                  f"기본 상한 {total}으로 균등 샘플링 시도")
        n = min(args.num_samples, total)
        indices = sorted(set(np.linspace(0, total - 1, n, dtype=int).tolist()))
    print(f"[frames] 처리할 프레임 인덱스: {indices}")

    frames = extract_frames_at_indices(sources, indices)
    if not frames:
        sys.exit("[err] 요청한 인덱스에서 프레임을 하나도 못 읽었음 (비디오 디코딩 실패)")

    try:
        estimator = build_traversability_estimator(args)  # 프레임 루프 밖에서 한 번만 생성(모델 재로딩 방지)
    except ImportError as e:
        sys.exit(f"[err] {e}")

    out_dir = os.path.join(args.out, f"ride_{rid}")
    saved = []
    for idx in indices:
        if idx not in frames:
            print(f"[skip] frame_idx={idx} 디코딩 실패")
            continue
        path = process_frame(frames[idx], idx, rid, out_dir, args, estimator)
        saved.append(path)
        print(f"[save] {path}")

    print(f"\n[done] {len(saved)}장 저장 완료 -> {out_dir}")


if __name__ == "__main__":
    main()
