import os
import cv2
import pickle
import random
import argparse
import numpy as np
from fastreid.emb_computer import EmbeddingComputer


def make_parser():
    # Initialization
    parser = argparse.ArgumentParser("Feature Extractor for Custom Dataset")

    # --- Data args (You will provide these) ---
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root path to the image split, e.g., /path/to/DynUAV/test")
    parser.add_argument("--pickle_path", type=str, required=True,
                        help="Path to the input detection pickle file, e.g., ./uav_detections_corrected.pickle")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the output pickle file with features, e.g., ./uav_detections_with_feats.pickle")
    
    # --- Model args (Usually fixed if using pre-trained model) ---
    parser.add_argument("--config_path", type=str, default="configs/MOT17/sbs_S50.yml",
                        help="Path to the FastReID model config file")
    parser.add_argument("--weight_path", type=str, default="weights/mot17_sbs_S50.pth",
                        help="Path to the FastReID model weights")
    parser.add_argument("--image_name_format", type=str, default="frame_{:06d}.jpg",
                        help="Python format string for image filenames, using zero-based frame index.")

    # --- Else ---
    parser.add_argument("--seed", type=float, default=10000)

    return parser


if __name__ == "__main__":
    # Get arguments
    args = make_parser().parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Get encoder
    print("Initializing feature embedding computer...")
    embedder = EmbeddingComputer(config_path=args.config_path, weight_path=args.weight_path)
    print("Model loaded.")

    # Read detection
    print(f"Loading detections from: {args.pickle_path}")
    with open(args.pickle_path, 'rb') as f:
        detections = pickle.load(f)

    # Feature extraction
    print("Starting feature extraction...")
    for vid_name in detections.keys():
        # Using a simple progress indicator
        num_frames = len(detections[vid_name])
        print(f"\nProcessing video: {vid_name} ({num_frames} frames)")
        
        for i, frame_id in enumerate(sorted(detections[vid_name].keys())):
            # If there is no detection for this frame, skip
            if detections[vid_name][frame_id] is None:
                continue

            # ======================== MODIFICATION FOR YOUR DATASET ========================
            # Original code had a complex check for MOT17/MOT20 filename padding.
            # We replace it with a direct path construction for your dataset structure.
            # DynUAV frame names are zero-based, for example frame_000000.jpg.
            img_filename = args.image_name_format.format(frame_id - 1)
            img_path = os.path.join(args.data_path, vid_name, 'img1', img_filename)
            # ===============================================================================

            # Read image
            img = cv2.imread(img_path)

            # Add an error check for safety
            if img is None:
                print(f"Warning: Could not read image at {img_path}. Skipping frame {frame_id} of video {vid_name}.")
                continue

            # Get detection
            detection = detections[vid_name][frame_id]

            # Get features
            if detection is not None and len(detection) > 0:
                embedding = embedder.compute_embedding(img, detection[:, :4])
                # Concatenate [detection_info, features]
                detections[vid_name][frame_id] = np.concatenate([detection, embedding], axis=1)

            # Simple progress print
            if (i + 1) % 100 == 0 or (i + 1) == num_frames:
                print(f"  ... processed frame {i + 1}/{num_frames}", flush=True)

    # Save the final results with features
    print(f"\nFeature extraction complete. Saving results to: {args.output_path}")
    with open(args.output_path, 'wb') as handle:
        pickle.dump(detections, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Done!")
