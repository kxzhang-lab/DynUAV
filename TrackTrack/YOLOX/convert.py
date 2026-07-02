import os
import cv2
import glob
import pickle
import numpy as np
import argparse
from tqdm import tqdm

def cxcywh_to_x1y1x2y2(box):
    """
    Converts absolute pixel coordinates from (center_x, center_y, width, height)
    to (x1, y1, x2, y2).
    """
    center_x, center_y, w, h = box
    
    x1 = center_x - w / 2
    y1 = center_y - h / 2
    x2 = center_x + w / 2
    y2 = center_y + h / 2
    
    return [x1, y1, x2, y2]

def process_detections(det_root, img_root, output_path):
    """
    Converts YOLO detection .txt files (with absolute pixel cxcywh format) 
    to a single pickle file required by TrackTrack.
    """
    det_results = {}
    
    video_dirs = sorted([d for d in os.listdir(det_root) if os.path.isdir(os.path.join(det_root, d))])
    
    print(f"Found {len(video_dirs)} video sequences...")

    for video_name in tqdm(video_dirs, desc="Processing videos"):
        det_results[video_name] = {}
        
        current_det_path = os.path.join(det_root, video_name)
        current_img_parent_path = os.path.join(img_root, video_name)
        
        det_files = sorted(glob.glob(os.path.join(current_det_path, '*.txt')))
        
        if not det_files:
            print(f"Warning: No detection files found for video {video_name}")
            continue

        for det_file in det_files:
            base_name = os.path.basename(det_file)
            frame_str = os.path.splitext(base_name)[0].split('_')[-1]
            frame_id = int(frame_str)
            frame_id += 1

            img_filename = os.path.splitext(base_name)[0] + '.jpg' 
            img_path = os.path.join(current_img_parent_path, 'img1', img_filename)

            if not os.path.exists(img_path):
                print(f"Error: Image file not found at {img_path}. Skipping frame {frame_id} of video {video_name}.")
                det_results[video_name][frame_id] = None
                continue
                
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape
            
            detections_in_frame = []
            with open(det_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 6:
                        continue
                        
                    class_id = int(float(parts[0]))
                    # These are already absolute pixel values
                    cx, cy, w, h = map(float, parts[1:5])
                    confidence = float(parts[5])
                    
                    # =================== FIX IS HERE ===================
                    # Use the new conversion function for absolute pixel values
                    x1, y1, x2, y2 = cxcywh_to_x1y1x2y2([cx, cy, w, h])
                    # ===================================================
                    
                    detections_in_frame.append([x1, y1, x2, y2, confidence, class_id])

            if detections_in_frame:
                final_dets = np.array(detections_in_frame, dtype=np.float32)
                # Clipping is still a good practice to handle edge cases
                final_dets[:, 0] = np.maximum(0, final_dets[:, 0])
                final_dets[:, 1] = np.maximum(0, final_dets[:, 1])
                final_dets[:, 2] = np.minimum(img_w - 1, final_dets[:, 2])
                final_dets[:, 3] = np.minimum(img_h - 1, final_dets[:, 3])
                det_results[video_name][frame_id] = final_dets
            else:
                det_results[video_name][frame_id] = None

    print(f"\nSaving detection results to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(det_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Conversion complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert YOLO detections to TrackTrack pickle format.")
    parser.add_argument('--det_root', type=str, required=True,
                        help="Path to the root directory containing YOLO detection folders (e.g., /path/to/detections/test).")
    parser.add_argument('--img_root', type=str, required=True,
                        help="Path to the root directory containing corresponding image folders (e.g., /path/to/ours/test).")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Path to save the final .pickle file.")
    
    args = parser.parse_args()
    
    process_detections(args.det_root, args.img_root, args.output_path)
    
