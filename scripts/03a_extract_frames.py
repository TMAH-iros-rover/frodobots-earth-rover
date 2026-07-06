"""
03a_extract_frames.py
front_camera mp4에서 프레임을 순차 디코딩해 리사이즈 저장하고,
obs_action_pairs.csv 와 합쳐 학습용 manifest.csv 를 만든다.
(학습 때 mp4를 매번 random seek 하면 매우 느리므로 미리 뽑아둔다.)

사용:
  python 03a_extract_frames.py ../data/frodobots-dataset-getting-started/ride_19154_20240225023555 \
      --pairs obs_action_pairs.csv --out train_data --size 224
"""
import os, sys, argparse
import cv2
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--pairs", default="obs_action_pairs.csv")
    ap.add_argument("--out", default="train_data")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    session_dir = os.path.expanduser(args.session_dir)
    rid = os.path.basename(session_dir.rstrip("/")).split("_")[1]
    mp4 = os.path.join(session_dir, f"front_camera_{rid}.mp4")

    pairs = pd.read_csv(args.pairs)
    n_pairs = len(pairs)
    frames_dir = os.path.join(args.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        sys.exit(f"[err] mp4 열기 실패: {mp4}")

    idx, saved, paths = 0, 0, {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx < n_pairs:
            img = cv2.resize(frame, (args.size, args.size))
            fp = os.path.join(frames_dir, f"frame_{idx:05d}.jpg")
            cv2.imwrite(fp, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            paths[idx] = fp
            saved += 1
        idx += 1
    cap.release()
    print(f"[frames] mp4 총 {idx} 프레임, 저장 {saved}장 (pairs {n_pairs}개)")

    # manifest = pairs 중 프레임이 실제로 뽑힌 것만
    pairs = pairs[pairs.frame_idx.isin(paths.keys())].copy()
    pairs["frame_path"] = pairs.frame_idx.map(paths)
    man_path = os.path.join(args.out, "manifest.csv")
    pairs.to_csv(man_path, index=False)
    print(f"[manifest] {len(pairs)}행 -> {man_path}")
    print(pairs[["frame_idx", "frame_path", "linear", "angular"]].head())


if __name__ == "__main__":
    main()
