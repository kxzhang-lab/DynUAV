import os
import cv2
import pickle
import numpy as np
import argparse
from tqdm import tqdm


def cxcywh_to_x1y1x2y2(box):
    """Convert from (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = box
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]


def process_detections(root_dir, output_path, conf_thres=0.0):
    """
    Convert all detection txt files in DHUAV dataset structure into a single pickle file.
    Each sequence folder must contain: det/det.txt and img1/.
    """
    det_results = {}

    seq_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    print(f"Found {len(seq_dirs)} video sequences: {seq_dirs}")

    for seq_name in tqdm(seq_dirs, desc="Processing sequences"):
        seq_path = os.path.join(root_dir, seq_name)
        det_file = os.path.join(seq_path, "det", "det.txt")
        img_folder = os.path.join(seq_path, "img1")

        if not os.path.exists(det_file):
            print(f"Warning: {det_file} was not found. Skipping this sequence.")
            continue
        if not os.path.exists(img_folder):
            print(f"Warning: {img_folder} was not found. Skipping this sequence.")
            continue

        det_results[seq_name] = {}

        with open(det_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue

            frame_id = int(parts[0])
            cx, cy, w, h = map(float, parts[2:6])
            conf = float(parts[6])
            if conf < conf_thres:
                continue
            class_id = 0

            x1, y1, x2, y2 = cxcywh_to_x1y1x2y2([cx, cy, w, h])

            img_path = os.path.join(img_folder, f"{frame_id:05d}.jpg")
            if not os.path.exists(img_path):
                det_results[seq_name][frame_id] = None
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            img_h, img_w, _ = img.shape

            # Clip to image bounds
            x1 = np.clip(x1, 0, img_w - 1)
            y1 = np.clip(y1, 0, img_h - 1)
            x2 = np.clip(x2, 0, img_w - 1)
            y2 = np.clip(y2, 0, img_h - 1)

            det = [x1, y1, x2, y2, conf, class_id]

            if frame_id not in det_results[seq_name]:
                det_results[seq_name][frame_id] = []
            det_results[seq_name][frame_id].append(det)

        for fid in det_results[seq_name]:
            if det_results[seq_name][fid] is not None:
                det_results[seq_name][fid] = np.array(det_results[seq_name][fid], dtype=np.float32)

    print(f"\nSaving detection results to {output_path} ...")
    with open(output_path, "wb") as f:
        pickle.dump(det_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DHUAV YOLO detections to TrackTrack pickle format.")
    parser.add_argument('--root_dir', type=str, required=True,
                        help="Dataset root path, e.g., /path/to/DHUAV")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Output pickle path")
    parser.add_argument('--conf_thres', type=float, default=0.0,
                        help="Drop detections below this confidence threshold")
    args = parser.parse_args()

    process_detections(args.root_dir, args.output_path, args.conf_thres)
