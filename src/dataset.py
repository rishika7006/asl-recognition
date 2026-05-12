"""PyTorch Dataset that lazily extracts flow features per epoch.

Augmentation is applied to **raw frames before flow is computed**, so the
classifier sees genuinely different flow distributions per epoch.

The Dataset operates on raw .mp4 files, doing
sample → augment → crop → flow on every __getitem__:

- No on-disk cache of cropped clips to maintain.
- The hand bbox is computed once from the un-augmented frames per video and
  reused, so MediaPipe runs once per video, not once per epoch.
- For val/test the feature vector is cached after first call so subsequent
  epochs are fast and deterministic.

Training-only augmentations:
    - Random horizontal flip (50%): u-channel of resulting flow is negated
      to stay consistent with the flipped image.
    - Brightness / contrast jitter (±15%).
    - Random rotation (±5°) + translation (±8 px), applied identically to
      all 16 frames so the motion stays coherent.
    - Temporal jitter: random offset within the action window of ±10% of
      the window length, plus ±10% stride variation.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.extract_features import (
    _read_frames_at,
    features_from_cropped_clip,
)
from src.flow_backends import get_backend
from src.hand_crop import HandCropper

_logger = logging.getLogger(__name__)


def _build_video_index(
    json_path: str,
    videos_dir: str,
    missing_txt: Optional[str],
    top_k: int,
    subset: str,
) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    """Build a list of dicts describing each clip in a subset.

    Each entry has: ``video_id``, ``video_path``, ``label``,
    ``orig_class``, ``start``, ``end``. Videos that don't exist on disk
    are filtered out. ``label`` is the remapped 0..K-1 class index.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    counts = Counter(info["action"][0] for info in data.values())
    top_classes = [c for c, _ in counts.most_common(top_k)]
    class_map = {orig: idx for idx, orig in enumerate(top_classes)}

    missing: set[str] = set()
    if missing_txt and os.path.exists(missing_txt):
        with open(missing_txt, "r") as f:
            missing = {line.strip() for line in f if line.strip()}

    out: List[Dict[str, Any]] = []
    for vid, info in data.items():
        if info["subset"] != subset:
            continue
        if vid in missing:
            continue
        if info["action"][0] not in top_classes:
            continue
        path = os.path.join(videos_dir, f"{vid}.mp4")
        if not os.path.exists(path):
            continue
        out.append(
            dict(
                video_id=vid,
                video_path=path,
                label=class_map[info["action"][0]],
                orig_class=info["action"][0],
                start=int(info["action"][1]),
                end=int(info["action"][2]),
            )
        )
    return out, class_map


def _open_capture(video_path: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def _augment_clip(
    frames: List[np.ndarray],
    flip: bool,
    angle_deg: float,
    tx: int,
    ty: int,
    brightness: float,
    contrast: float,
) -> List[np.ndarray]:
    """Apply the same affine + photometric perturbation to every frame.

    Args:
        frames: list of ``(H, W, 3)`` uint8 RGB.
        flip: horizontal flip if True.
        angle_deg, tx, ty: rotation (around center) and translation.
        brightness: additive shift in [-1, 1] (rescaled to ±255).
        contrast: multiplicative gain factor (e.g. 0.85..1.15).

    Returns:
        New list of (H, W, 3) uint8 RGB frames.
    """
    if not frames:
        return frames
    h, w = frames[0].shape[:2]
    # Single rotation+translation matrix reused for the whole clip so motion stays coherent.
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty

    bright_offset = np.float32(brightness * 255.0)
    contrast_gain = np.float32(contrast)

    out: List[np.ndarray] = []
    for f in frames:
        if flip:
            f = cv2.flip(f, 1)
        f = cv2.warpAffine(
            f,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        f32 = f.astype(np.float32) * contrast_gain + bright_offset
        out.append(np.clip(f32, 0, 255).astype(np.uint8))
    return out


def _negate_u_channel(features: np.ndarray, grid_size: int) -> np.ndarray:
    """Negate the u (horizontal) component of a pooled-flow feature stack.

    Layout from ``grid_pool_flow`` is ``[u0, v0, u1, v1, ...]`` per pair,
    so u entries live at every even index along the last axis.
    """
    if features.ndim != 2:
        raise ValueError(f"expected (T-1, F) features, got {features.shape}")
    expected = grid_size * grid_size * 2
    if features.shape[1] != expected:
        raise ValueError(
            f"features last dim {features.shape[1]} != grid_size**2 * 2 = {expected}"
        )
    out = features.copy()
    out[:, 0::2] = -out[:, 0::2]
    return out


class ClipFeatureDataset(Dataset):
    """PyTorch Dataset of optical-flow feature stacks for WLASL clips.

    For ``subset='train'`` the features are recomputed every
    ``__getitem__`` call with frame-level augmentation. For
    ``subset='val'`` and ``'test'`` features are computed once and
    cached.

    Args:
        config: parsed YAML config.
        subset: ``"train"``, ``"val"``, or ``"test"``.
        backend_name: ``"farneback"`` or ``"raft"``.
        device: device for backends that need it (RAFT only).
        augment: enable training-time augmentation. Default: True iff
            ``subset == "train"``.
        cropper: optional shared HandCropper. If None, one is created.
    """

    def __init__(
        self,
        config: Dict,
        subset: str,
        backend_name: str = "farneback",
        device: str = "cpu",
        augment: Optional[bool] = None,
        cropper: Optional[HandCropper] = None,
    ) -> None:
        self.config = config
        self.subset = subset
        ds_cfg = config["dataset"]
        self._json_path = ds_cfg["json_path"]
        self._videos_dir = ds_cfg["videos_dir"]
        self._missing_txt = ds_cfg.get("missing_txt")
        self._top_k = int(ds_cfg["top_k"])
        self._num_frames = int(ds_cfg["num_frames"])
        self._crop_size = int(ds_cfg.get("crop_size", 96))
        self._grid_size = int(ds_cfg.get("grid_size", 12))

        self._backend = get_backend(backend_name)
        self._estimator = self._backend.make_estimator(device=device)
        self._owns_cropper = cropper is None
        self._cropper = cropper or HandCropper()

        self._augment = bool(augment) if augment is not None else (subset == "train")

        self.entries, self.class_map = _build_video_index(
            self._json_path,
            self._videos_dir,
            self._missing_txt,
            self._top_k,
            subset,
        )

        # MediaPipe runs once per video, not once per epoch.
        self._bbox_cache: Dict[str, Optional[Tuple[int, int, int, int]]] = {}
        self._feat_cache: Dict[str, np.ndarray] = {}

    @property
    def num_classes(self) -> int:
        return len(self.class_map)

    @property
    def feature_shape(self) -> Tuple[int, int]:
        return (self._num_frames - 1, self._grid_size * self._grid_size * 2)

    def __len__(self) -> int:
        return len(self.entries)

    def _get_or_compute_bbox(
        self, video_id: str, frames: List[np.ndarray]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Lookup-or-compute the per-clip bbox from un-augmented frames."""
        if video_id in self._bbox_cache:
            return self._bbox_cache[video_id]
        bbox = self._cropper.bbox_for_clip(frames)
        self._bbox_cache[video_id] = bbox
        return bbox

    def _sample_indices(self, total: int, start: int, end: int) -> List[int]:
        """Pick frame indices in the action window, with optional jitter."""
        last = total - 1
        s = max(0, min(start, last))
        e = max(0, min(end, last))
        if e <= s:
            s, e = 0, last
        if e == s:
            return [s] * self._num_frames
        if self._augment:
            window = e - s
            offset_max = max(1, int(0.10 * window))
            stride_jitter = 0.10
            shift = random.randint(-offset_max, offset_max)
            scale = 1.0 + random.uniform(-stride_jitter, stride_jitter)
            mid = 0.5 * (s + e) + shift
            half = 0.5 * window * scale
            new_s = max(0, mid - half)
            new_e = min(last, mid + half)
            if new_e <= new_s:
                new_s, new_e = float(s), float(e)
            return list(np.linspace(new_s, new_e, self._num_frames, dtype=int))
        return list(np.linspace(s, e, self._num_frames, dtype=int))

    def _read_clip(self, entry: Dict[str, Any]) -> Optional[List[np.ndarray]]:
        cap = _open_capture(entry["video_path"])
        if cap is None:
            return None
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 1:
                return None
            indices = self._sample_indices(total, entry["start"], entry["end"])
            frames = _read_frames_at(cap, indices)
        finally:
            cap.release()
        if not frames:
            return None
        while len(frames) < self._num_frames:
            frames.append(frames[-1])
        return frames

    def _features_for_entry(self, entry: Dict[str, Any]) -> np.ndarray:
        """Compute (or retrieve) the feature stack for one entry."""
        if not self._augment and entry["video_id"] in self._feat_cache:
            return self._feat_cache[entry["video_id"]]

        frames = self._read_clip(entry)
        if frames is None:
            # Defensive fallback; _build_video_index already filters missing files.
            return np.zeros(self.feature_shape, dtype=np.float32)

        flip = False
        if self._augment:
            flip = random.random() < 0.5
            angle = random.uniform(-5.0, 5.0)
            tx = random.randint(-8, 8)
            ty = random.randint(-8, 8)
            brightness = random.uniform(-0.15, 0.15)
            contrast = random.uniform(0.85, 1.15)
            aug_frames = _augment_clip(
                frames,
                flip=flip,
                angle_deg=angle,
                tx=tx,
                ty=ty,
                brightness=brightness,
                contrast=contrast,
            )
        else:
            aug_frames = frames

        bbox = self._get_or_compute_bbox(entry["video_id"], frames)
        # The cached bbox is in unflipped coords; flip it horizontally to
        # match the augmented frames.
        if flip and bbox is not None:
            h, w = aug_frames[0].shape[:2]
            x1, y1, x2, y2 = bbox
            bbox = (w - x2, y1, w - x1, y2)

        cropped = self._cropper.crop_clip(aug_frames, bbox, self._crop_size)
        if len(cropped) < 2:
            return np.zeros(self.feature_shape, dtype=np.float32)
        feats = features_from_cropped_clip(
            cropped, self._backend, self._estimator, self._grid_size
        )
        if flip:
            # Negate u-channel so post-flip flow matches what would be produced
            # from the un-flipped frames.
            feats = _negate_u_channel(feats, self._grid_size)

        if not self._augment:
            self._feat_cache[entry["video_id"]] = feats
        return feats

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        entry = self.entries[idx]
        feats = self._features_for_entry(entry)
        return torch.from_numpy(feats), int(entry["label"])

    def close(self) -> None:
        if self._owns_cropper:
            try:
                self._cropper.close()
            except Exception:  # noqa: BLE001
                pass

    def __del__(self) -> None:  # noqa: D401
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
