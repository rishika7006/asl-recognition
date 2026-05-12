import os
import json
import glob
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

# We'll import these to compute features on the fly and cache them
from src.optical_flow import extract_flow_sequence
from src.hand_landmarks import HandLandmarkExtractor

def get_top_k_classes(json_path, k=10):
    with open(json_path, 'r') as f:
        data = json.load(f)
    class_counts = Counter()
    for video_id, info in data.items():
        class_id = info['action'][0]
        class_counts[class_id] += 1
    top_k = class_counts.most_common(k)
    return [c[0] for c in top_k]

def get_data_splits(json_path, top_k_classes, missing_txt_path=None):
    with open(json_path, 'r') as f:
        data = json.load(f)
    missing_videos = set()
    if missing_txt_path and os.path.exists(missing_txt_path):
        with open(missing_txt_path, 'r') as f:
            missing_videos = set([line.strip() for line in f.readlines()])
    splits = {'train': [], 'val': [], 'test': []}
    class_map = {orig_class: idx for idx, orig_class in enumerate(top_k_classes)}
    for video_id, info in data.items():
        if video_id in missing_videos:
            continue
        class_id = info['action'][0]
        if class_id in top_k_classes:
            subset = info['subset']
            if subset in splits:
                splits[subset].append({'video_id': video_id, 'class_id': class_map[class_id]})
    return splits, class_map

class ASLDataset(Dataset):
    def __init__(self, data_list, frames_dir, num_frames=16, transform=None):
        self.data_list = data_list
        self.frames_dir = frames_dir
        self.num_frames = num_frames
        self.transform = transform
        self.cache_dir = os.path.join(os.path.dirname(frames_dir), 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.lm_extractor = HandLandmarkExtractor()

    def __len__(self):
        return len(self.data_list)
        
    def _load_frames(self, video_id):
        video_dir = os.path.join(self.frames_dir, video_id)
        if not os.path.exists(video_dir):
            return None
        frame_files = sorted(glob.glob(os.path.join(video_dir, '*.jpg')) + glob.glob(os.path.join(video_dir, '*.png')))
        if not frame_files:
            return None
        indices = np.linspace(0, len(frame_files) - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            frame = cv2.imread(frame_files[idx])
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
        if not frames:
            return None
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        return np.array(frames)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        video_id = item['video_id']
        label = item['class_id']
        
        # Paths for cached features
        flow_path = os.path.join(self.cache_dir, f"{video_id}_flow.pt")
        lm_path = os.path.join(self.cache_dir, f"{video_id}_lm.pt")
        rgb_path = os.path.join(self.cache_dir, f"{video_id}_rgb.pt")

        if os.path.exists(flow_path) and os.path.exists(lm_path) and os.path.exists(rgb_path):
            rgb = torch.load(rgb_path)
            flow = torch.load(flow_path)
            lm = torch.load(lm_path)
            return rgb, flow, lm, label, video_id

        frames_np = self._load_frames(video_id)
        if frames_np is None:
            rgb = torch.zeros((self.num_frames, 3, 224, 224))
            flow = torch.zeros((self.num_frames-1, 2, 224, 224))
            lm = torch.zeros((self.num_frames, 126))
            return rgb, flow, lm, label, video_id

        # Compute RGB tensor
        rgb = torch.from_numpy(frames_np).float() / 255.0
        rgb = rgb.permute(0, 3, 1, 2) # (T, C, H, W)
        
        # Compute Flow
        flow = extract_flow_sequence(rgb)
        
        # Compute Landmarks
        lm = self.lm_extractor.extract_sequence(rgb)

        # Cache
        torch.save(rgb, rgb_path)
        torch.save(flow, flow_path)
        torch.save(lm, lm_path)
        
        return rgb, flow, lm, label, video_id

def get_dataloaders(config, transform=None):
    json_path = config['dataset']['json_path']
    frames_dir = config['dataset']['frames_dir']
    missing_txt = config['dataset'].get('missing_txt', None)
    top_k = config['dataset']['top_k']
    num_frames = config['dataset']['num_frames']
    batch_size = config['training']['batch_size']
    
    top_k_classes = get_top_k_classes(json_path, k=top_k)
    splits, class_map = get_data_splits(json_path, top_k_classes, missing_txt)
    
    dataloaders = {}
    for subset in ['train', 'val', 'test']:
        dataset = ASLDataset(splits[subset], frames_dir, num_frames=num_frames, transform=transform)
        shuffle = (subset == 'train')
        dataloaders[subset] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloaders, class_map
