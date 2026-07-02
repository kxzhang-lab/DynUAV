import os
import cv2
import glob
import time
import torch
import pickle
import argparse
import random
import numpy as np
from utils.etc import *
from AFLink.AppFreeLink import *
from AFLink.model import PostLinker
from AFLink.dataset import LinkData
from trackers.tracker import Tracker
from utils.gbi import gb_interpolation

def make_parser():
    parser = argparse.ArgumentParser("Custom Tracker Runner")

    # --- Paths (Required) ---
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root path to the image split, e.g., /path/to/DynUAV/test")
    parser.add_argument("--pickle_path", type=str, required=True,
                        help="Path to the detection+feature pickle file (lower confidence threshold)")
    parser.add_argument("--pickle_path_95", type=str, required=True,
                        help="Path to the high-confidence detection+feature pickle file. Can be the same as --pickle_path to start.")
    parser.add_argument("--output_dir", type=str, default="../outputs/3. track/",
                        help="Directory to save tracking results")

    # --- Configs (Provide a name for your dataset) ---
    parser.add_argument("--dataset", type=str, default="UAV_Custom", help="A name for your dataset")
    parser.add_argument("--mode", type=str, default="test", help="Should be 'test' to avoid evaluation")
    parser.add_argument("--vid_names", type=str, default=None,
                        help="Comma-separated sequence names. If omitted, all sequences in the pickle are used.")
    parser.add_argument("--seed", type=float, default=10000)

    # --- Post-Processing ---
    parser.add_argument("--post_process", action='store_true', 
                        help="Enable Gaussian-based Interpolation as post-processing.")

    # --- Tracker Hyperparameters (can be tuned) ---
    parser.add_argument("--min_len", type=int, default=3)
    parser.add_argument("--min_box_area", type=float, default=100)
    parser.add_argument("--default_fps", type=int, default=30, help="Default FPS if seqinfo.ini is not found")
    # max_time_lost will be set based on FPS later
    parser.add_argument("--penalty_p", type=float, default=0.20)
    parser.add_argument("--penalty_q", type=float, default=0.40)
    parser.add_argument("--reduce_step", type=float, default=0.05)
    parser.add_argument("--tai_thr", type=float, default=0.55)

    return parser

def track(detections, detections_95, data_path, result_folder, mode, video_names=None):
    total_time, total_count = 0, 0
    vid_names = video_names if video_names is not None else sorted(detections.keys())
    for vid_name in vid_names:
        if vid_name not in detections:
            print(f"Warning: video {vid_name} not found in detection pickle. Skipping.")
            continue
        print(f"\n--- Processing video: {vid_name} ---")
        
        # ==================== MODIFICATION: GET IMG_SIZE & FPS DYNAMICALLY ====================
        # Try to find seqinfo.ini for compatibility, otherwise get info from image
        seq_info_path = os.path.join(data_path, vid_name, 'seqinfo.ini')
        print(f"Looking for seqinfo at: {seq_info_path}")
        if os.path.exists(seq_info_path):
            seq_info = open(seq_info_path, mode='r')
            for s_i in seq_info.readlines():
                if 'frameRate' in s_i:
                    fps = int(s_i.split('=')[-1])
                if 'imWidth' in s_i:
                    args.img_w = int(s_i.split('=')[-1])
                if 'imHeight' in s_i:
                    args.img_h = int(s_i.split('=')[-1])
            args.max_time_lost = 30 * 2 # Set max lost time to 2 seconds
        else:
            # Fallback: read the first image to get dimensions
            # We assume the image path structure is data_path/vid_name/img1/*.jpg
            first_image_path_pattern = os.path.join(data_path, vid_name, 'img1', '*.jpg')
            image_files = sorted(glob.glob(first_image_path_pattern))
            if not image_files:
                raise FileNotFoundError(f"No images found for video {vid_name} at {first_image_path_pattern}")
            
            first_img = cv2.imread(image_files[0])
            args.img_h, args.img_w, _ = first_img.shape
            fps = args.default_fps
            args.max_time_lost = fps * 2
            print(f"'{seq_info_path}' not found. Inferred image size: ({args.img_w}x{args.img_h}), using default FPS: {fps}")
        # ======================================================================================

        tracker = Tracker(args, vid_name)

        results = []
        num_frames = len(detections[vid_name])
        for i, frame_id in enumerate(sorted(detections[vid_name].keys())):
            start = time.time()
            if detections[vid_name][frame_id] is not None:
                track_results = tracker.update(detections[vid_name][frame_id], detections_95[vid_name].get(frame_id))
            else:
                track_results = tracker.update_without_detections()
            total_time += time.time() - start
            total_count += 1
            
            # Simple progress print
            if (i + 1) % 100 == 0 or (i + 1) == num_frames:
                print(f"  ... processed frame {i + 1}/{num_frames}", flush=True)

            x1y1whs, track_ids, scores = [], [], []
            for t in track_results:
                # ==================== MODIFICATION: REMOVED ASPECT RATIO FILTER ===================
                # This filter was specific to pedestrians in MOT datasets.
                # if 'MOT' in data_path and t.x1y1wh[2] / t.x1y1wh[3] > 1.6:
                #     continue
                # ================================================================================

                if t.track_id > 0 and t.x1y1wh[2] * t.x1y1wh[3] > args.min_box_area:
                    x1y1whs.append(t.x1y1wh)
                    track_ids.append(t.track_id)
                    scores.append(t.score)

            # results.append([frame_id, track_ids, x1y1whs, scores])
            results.append([frame_id-1, track_ids, x1y1whs, scores])

        result_filename = os.path.join(result_folder, '{}.txt'.format(vid_name))
        write_results(result_filename, results)
        print(f"Tracking for {vid_name} complete. Results saved to {result_filename}")

    return total_time, total_count

def run():
    # Logging & Set tracker parameters for the DynUAV custom entrypoint.
    print(f'Running tracking for dataset "{args.dataset}" in "{args.mode}" mode.')
    args.det_thr = 0.60
    args.init_thr = 0.70
    args.match_thr = 0.70
    args.data_path = args.data_dir

    # Make result folder
    pickle_name = os.path.basename(args.pickle_path).replace('.pickle', '')
    result_folder = os.path.join(args.output_dir, pickle_name)
    os.makedirs(result_folder, exist_ok=True)
    
    post_result_folder = os.path.join(args.output_dir, pickle_name + '_post')
    if args.post_process:
        os.makedirs(post_result_folder, exist_ok=True)

    # Read detection result
    print(f"Loading main detections from: {args.pickle_path}")
    with open(args.pickle_path, 'rb') as f:
        detections = pickle.load(f)
    if os.path.abspath(args.pickle_path) == os.path.abspath(args.pickle_path_95):
        detections_95 = detections
    else:
        print(f"Loading high-confidence detections from: {args.pickle_path_95}")
        with open(args.pickle_path_95, 'rb') as f:
            detections_95 = pickle.load(f)

    # Track
    video_names = None
    if args.vid_names:
        video_names = [name.strip() for name in args.vid_names.split(',') if name.strip()]

    total_time, total_count = track(detections, detections_95, args.data_dir, result_folder, args.mode, video_names)

    # Post-processing
    if args.post_process:
        print('Running post-processing (Gaussian Interpolation)...')
        for result_file in os.listdir(result_folder):
            path_in = os.path.join(result_folder, result_file)
            path_out = os.path.join(post_result_folder, result_file)
            gb_interpolation(path_in, path_out, interval=30, tau=12)

    if total_time > 0:
        print(f"\nAverage tracking speed: {total_count / total_time:.2f} FPS", flush=True)
    print("Tracking finished!", flush=True)

if __name__ == "__main__":
    args = make_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    run()
