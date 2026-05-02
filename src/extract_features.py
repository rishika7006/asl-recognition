import os
import cv2
import json
import yaml
import glob
import numpy as np
from collections import Counter
from tqdm import tqdm
from src.hand_landmarks import HandLandmarkExtractor


def extract_combined_features(frames, extractor, grid_size=8):
    """
    Extracts both Farneback Optical Flow and MediaPipe hand landmarks per frame.
    Since optical flow needs 2 frames, it returns T-1 frames.
    frames: list of numpy arrays (H, W, 3) in RGB
    Returns: 2D numpy array of size (len(frames)-1, 126 + 128)
    """
    seq_features = []
    
    # 1. Get MediaPipe landmarks for all frames
    all_landmarks = []
    for frame in frames:
        landmarks = extractor.extract_from_frame(frame)
        all_landmarks.append(landmarks)
        
    # 2. Get Farneback flow
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    h, w = prev_gray.shape
    cell_h = h // grid_size
    cell_w = w // grid_size

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Grid-based mean pooling
        flow_features = []
        for r in range(grid_size):
            for c in range(grid_size):
                cell_flow = flow[
                    r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w
                ]
                mean_u = np.mean(cell_flow[..., 0])
                mean_v = np.mean(cell_flow[..., 1])
                flow_features.extend([mean_u, mean_v])
                
        prev_gray = curr_gray
        
        # Combine Flow (128) + Landmarks (126) for this frame step
        # We use the landmarks of the *current* frame (index i)
        combined = np.concatenate([flow_features, all_landmarks[i]])
        seq_features.append(combined)

    return np.array(seq_features, dtype=np.float32)


def augment_sequence(seq):
    """
    Adds small random Gaussian noise to simulate slightly different movements/positions.
    """
    noise = np.random.normal(0, 0.02, seq.shape).astype(np.float32)
    return seq + noise


def main():
    with open("configs/top10.yaml", "r") as f:
        config = yaml.safe_load(f)

    json_path = config["dataset"]["json_path"]
    frames_dir = config["dataset"]["frames_dir"]
    missing_txt = config["dataset"].get("missing_txt", None)
    top_k = config["dataset"]["top_k"]
    num_frames = config["dataset"]["num_frames"]

    with open(json_path, "r") as f:
        data = json.load(f)

    class_counts = Counter()
    for vid, info in data.items():
        class_counts[info["action"][0]] += 1

    top_classes = [c[0] for c in class_counts.most_common(top_k)]
    class_map = {orig_class: idx for idx, orig_class in enumerate(top_classes)}

    missing_videos = set()
    if missing_txt and os.path.exists(missing_txt):
        with open(missing_txt, "r") as f:
            missing_videos = set([line.strip() for line in f.readlines()])

    X_dict = {"train": [], "val": [], "test": []}
    y_dict = {"train": [], "val": [], "test": []}

    video_list = [
        (vid, info)
        for vid, info in data.items()
        if vid not in missing_videos and info["action"][0] in top_classes
    ]

    print(f"Extracting features for {len(video_list)} videos...")
    
    extractor = HandLandmarkExtractor()

    for vid, info in tqdm(video_list):
        subset = info["subset"]
        class_id = class_map[info["action"][0]]

        video_dir = os.path.join(frames_dir, vid)
        if not os.path.exists(video_dir):
            continue

        frame_files = sorted(
            glob.glob(os.path.join(video_dir, "*.jpg"))
            + glob.glob(os.path.join(video_dir, "*.png"))
        )
        if not frame_files:
            continue

        indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            frame = cv2.imread(frame_files[idx])
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (128, 128)) # Ensure constant size for flow
                frames.append(frame)

        if not frames:
            continue

        while len(frames) < num_frames:
            frames.append(frames[-1])

        seq_features = extract_combined_features(frames, extractor, grid_size=8)

        if subset in X_dict:
            # Original
            X_dict[subset].append(seq_features)
            y_dict[subset].append(class_id)
            
            # Augment training data to reduce overfitting
            if subset == "train":
                for _ in range(3): # 3 augmented copies
                    X_dict[subset].append(augment_sequence(seq_features))
                    y_dict[subset].append(class_id)

    os.makedirs("data/features", exist_ok=True)
    for subset in ["train", "val", "test"]:
        np.save(f"data/features/X_{subset}.npy", np.array(X_dict[subset]))
        np.save(f"data/features/y_{subset}.npy", np.array(y_dict[subset]))

    with open("data/features/class_map.json", "w") as f:
        json.dump(class_map, f)

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
