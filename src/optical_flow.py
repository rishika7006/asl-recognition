import cv2
import numpy as np
import torch
import os

def compute_dense_optical_flow(prev_frame, next_frame):
    """
    Computes Farneback dense optical flow.
    Args:
        prev_frame: numpy array (H, W, C) or (H, W)
        next_frame: numpy array (H, W, C) or (H, W)
    Returns:
        flow: numpy array (H, W, 2)
    """
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    else:
        prev_gray = prev_frame
        
    if len(next_frame.shape) == 3:
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
    else:
        next_gray = next_frame
        
    # Convert to 8-bit if necessary
    if prev_gray.dtype != np.uint8:
        prev_gray = (prev_gray * 255).astype(np.uint8)
    if next_gray.dtype != np.uint8:
        next_gray = (next_gray * 255).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow

def extract_flow_sequence(frames):
    """
    Extracts flow sequence from a sequence of frames.
    Args:
        frames: torch Tensor (T, C, H, W)
    Returns:
        flow_sequence: torch Tensor (T-1, 2, H, W)
    """
    # Convert back to numpy (T, H, W, C) for OpenCV
    frames_np = frames.permute(0, 2, 3, 1).numpy()
    T = frames_np.shape[0]
    
    flow_seq = []
    for t in range(T - 1):
        flow = compute_dense_optical_flow(frames_np[t], frames_np[t+1])
        flow_seq.append(flow)
        
    # Convert to tensor (T-1, 2, H, W)
    flow_seq = torch.from_numpy(np.array(flow_seq)).permute(0, 3, 1, 2)
    return flow_seq

