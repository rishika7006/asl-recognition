"""Render a Farnebäck-vs-RAFT flow visualization for the report.

Picks one user_videos clip (by default ``data/user_videos/computer/
computer_good_light.mov``), samples 4 evenly-spaced frame pairs, runs
both flow estimators, and writes a single 4×3 figure to
``eval_results/flow_visualization.png``:

    column 0: hand-cropped frame (the input the classifier sees)
    column 1: Farnebäck flow rendered as HSV (hue = direction, value = magnitude)
    column 2: RAFT flow, same encoding

Usage:
    python -m src.eval.viz_flow [--clip path/to/clip.mov] [--out eval_results/flow_visualization.png]
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.extract_features import _read_frames_at, _sample_indices_in_window
from src.flow_backends import get_backend
from src.hand_crop import HandCropper

OUT_DEFAULT = "eval_results/flow_visualization.png"
DEFAULT_CLIP = "data/user_videos/computer/computer_good_light.mov"
NUM_PAIRS_TO_SHOW = 4
CROP_SIZE = 96


def _flow_to_color(flow: np.ndarray) -> np.ndarray:
    """Render an (H, W, 2) flow field as an HSV-encoded RGB image.

    Hue = flow direction, value = magnitude (clipped). Saturation = 1.
    Returns (H, W, 3) uint8 RGB.
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    hsv[..., 0] = (ang * 180 / (2 * np.pi)).astype(np.uint8)
    hsv[..., 1] = 255
    # Cap value at p95 so a single large outlier doesn't black out everything else.
    cap = max(np.percentile(mag, 95), 1e-3)
    v = np.clip(mag / cap * 255, 0, 255).astype(np.uint8)
    hsv[..., 2] = v
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def _render(
    clip_path: str,
    out_path: str,
    num_pairs: int = NUM_PAIRS_TO_SHOW,
    crop_size: int = CROP_SIZE,
) -> None:
    if not os.path.exists(clip_path):
        raise FileNotFoundError(clip_path)

    cap = cv2.VideoCapture(clip_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    if total <= 1:
        cap.release()
        raise RuntimeError(f"Clip has no readable frames: {clip_path}")

    indices = _sample_indices_in_window(total, 0, total - 1, num_pairs + 1)
    frames = _read_frames_at(cap, indices)
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"Could not read enough frames from {clip_path}")

    cropper = HandCropper()
    bbox = cropper.bbox_for_clip(frames)
    crops = cropper.crop_clip(frames, bbox, target_size=crop_size)
    cropper.close()

    fb = get_backend("farneback")
    fb_est = fb.make_estimator(device="cpu")
    raft = get_backend("raft")
    raft_device = "mps" if torch.backends.mps.is_available() else "cpu"
    raft_est = raft.make_estimator(device=raft_device)

    n_pairs = min(num_pairs, len(crops) - 1)
    fig, axes = plt.subplots(n_pairs, 3, figsize=(9, 3 * n_pairs))
    if n_pairs == 1:
        axes = np.array([axes])

    for i in range(n_pairs):
        prev = crops[i]
        nxt = crops[i + 1]
        fb_flow = fb.flow(fb_est, prev, nxt)
        raft_flow = raft.flow(raft_est, prev, nxt)

        ax_img, ax_fb, ax_raft = axes[i, 0], axes[i, 1], axes[i, 2]
        ax_img.imshow(nxt)
        ax_img.set_title(f"hand crop (frame pair {i+1}/{n_pairs})", fontsize=10)
        ax_img.axis("off")

        ax_fb.imshow(_flow_to_color(fb_flow))
        ax_fb.set_title(
            f"Farnebäck flow  (|u| mean={np.abs(fb_flow[..., 0]).mean():.2f}, "
            f"|v| mean={np.abs(fb_flow[..., 1]).mean():.2f})",
            fontsize=9,
        )
        ax_fb.axis("off")

        ax_raft.imshow(_flow_to_color(raft_flow))
        ax_raft.set_title(
            f"RAFT flow  (|u| mean={np.abs(raft_flow[..., 0]).mean():.2f}, "
            f"|v| mean={np.abs(raft_flow[..., 1]).mean():.2f})",
            fontsize=9,
        )
        ax_raft.axis("off")

    fig.suptitle(
        f"Optical-flow comparison on {os.path.basename(clip_path)} — hue = direction, brightness = magnitude",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Farnebäck vs RAFT flow on a user clip."
    )
    parser.add_argument("--clip", default=DEFAULT_CLIP)
    parser.add_argument("--out", default=OUT_DEFAULT)
    parser.add_argument("--num-pairs", type=int, default=NUM_PAIRS_TO_SHOW)
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    args = parser.parse_args()
    _render(args.clip, args.out, args.num_pairs, args.crop_size)


if __name__ == "__main__":
    main()
