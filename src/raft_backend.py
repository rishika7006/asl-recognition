"""RAFT optical-flow backend for the ASL pipeline.

Conforms to the shared backend interface used alongside ``farneback_backend``:

    NAME: str
    make_estimator(device: str) -> opaque
    flow(estimator, prev_rgb, next_rgb) -> (H, W, 2) float32

The estimator is constructed once (model loaded onto the requested device,
weights frozen, eval mode) and re-used for every pair. Inference runs under
``torch.inference_mode()`` so the path is safe to call from a background
worker thread alongside Farnebäck.

Per the project plan (Phase 4/5), we run RAFT with **6 flow updates** instead
of the torchvision default of 12. This is roughly a 2x speedup with negligible
quality degradation given that downstream we mean-pool flow into a 12x12
grid anyway. The same iteration count is used for both offline feature
extraction (on Colab GPU) and live inference, to keep the flow distribution
the classifier sees consistent.

RAFT has two spatial constraints: dims must be divisible by 8 (because of
the /8 feature pyramid), AND each spatial dim must be at least 128 (the
correlation pyramid further downsamples and needs feature maps >= 16x16,
i.e. inputs >= 128). Our hand crops are 96x96, which violates the second
constraint, so we pad to the larger of (128, next multiple of 8) on each
axis with edge replication and crop the resulting flow back to the original
size before returning.
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

NAME: str = "raft"

# Number of recurrent flow-update iterations the RAFT decoder runs at
# inference. Default in torchvision is 12; we drop to 6 for ~2x speedup.
_NUM_FLOW_UPDATES: int = 6

# RAFT's feature pyramid downsamples by 8, so H and W must be multiples of 8.
_DIVISOR: int = 8

# RAFT's correlation pyramid further downsamples; the feature map after /8
# must be at least 16 in each spatial dim, so the *input* must be at least 128
# in each spatial dim. (torchvision raises a ValueError below 128.)
_MIN_SIDE: int = 128


def _resolve_device(device: str) -> torch.device:
    """Map a string device spec to a ``torch.device``.

    Accepts ``"cpu"``, ``"cuda"``, ``"mps"``. Any other value is passed
    through to ``torch.device`` directly so callers can provide e.g.
    ``"cuda:0"``.
    """
    return torch.device(device)


def make_estimator(device: str = "cpu") -> Dict[str, Any]:
    """Construct a stateful RAFT estimator on the given device.

    Args:
        device: ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns:
        A dict with keys ``model`` (eval-mode ``raft_small`` on the requested
        device), ``transforms`` (the torchvision preprocessing transform that
        normalizes inputs to [-1, 1]), and ``device``. This object is opaque
        and should be passed straight back into :func:`flow`.
    """
    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    torch_device = _resolve_device(device)

    model = raft_small(weights=weights, progress=False)
    model = model.to(torch_device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return {
        "model": model,
        "transforms": transforms,
        "device": torch_device,
    }


def _target_side(side: int) -> int:
    """Smallest H or W >= max(side, _MIN_SIDE) that is a multiple of _DIVISOR.

    For example, 96 -> 128, 100 -> 128, 128 -> 128, 130 -> 136, 64 -> 128.
    """
    target = max(side, _MIN_SIDE)
    if target % _DIVISOR != 0:
        target += _DIVISOR - (target % _DIVISOR)
    return target


def _pad_to_target(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Pad an (H, W, 3) uint8 image with edge replication so H and W satisfy
    RAFT's input constraints (multiple of 8 and at least 128).

    Pads only on the bottom and right; this keeps original pixels at indices
    [0:H, 0:W], so cropping the flow output back is a simple slice.

    Returns ``(padded, pad_bottom, pad_right)``.
    """
    h, w = img.shape[:2]
    target_h = _target_side(h)
    target_w = _target_side(w)
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h == 0 and pad_w == 0:
        return img, 0, 0
    padded = cv2.copyMakeBorder(
        img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REPLICATE
    )
    return padded, pad_h, pad_w


def _to_chw_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert an (H, W, 3) uint8 RGB array to a (1, 3, H, W) float32 tensor
    on the given device, scaled to [0, 1]. Subsequent normalization to
    [-1, 1] is performed by the RAFT preprocessing transform.
    """
    # Copy so we never alias the caller's array, and so PyTorch owns the memory.
    arr = np.ascontiguousarray(img)
    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    t = t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    t = t / 255.0
    return t


def flow(
    estimator: Dict[str, Any],
    prev_rgb: np.ndarray,
    next_rgb: np.ndarray,
) -> np.ndarray:
    """Compute RAFT optical flow between two RGB uint8 frames.

    Args:
        estimator: object returned by :func:`make_estimator`.
        prev_rgb: (H, W, 3) uint8 RGB.
        next_rgb: (H, W, 3) uint8 RGB, same H, W as ``prev_rgb``.

    Returns:
        (H, W, 2) float32 displacement field at the same H, W as the input.
        Channel 0 is horizontal flow (u), channel 1 is vertical flow (v).
    """
    if prev_rgb.shape != next_rgb.shape:
        raise ValueError(
            f"prev_rgb and next_rgb must have the same shape; "
            f"got {prev_rgb.shape} vs {next_rgb.shape}"
        )
    if prev_rgb.ndim != 3 or prev_rgb.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB; got shape {prev_rgb.shape}")
    if prev_rgb.dtype != np.uint8 or next_rgb.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 RGB; got dtypes {prev_rgb.dtype}, {next_rgb.dtype}"
        )

    h, w = prev_rgb.shape[:2]

    # Pad to RAFT's input constraints (multiple of 8 and >= 128 per side).
    # Operates on copies — does not mutate caller arrays.
    prev_padded, _pad_h, _pad_w = _pad_to_target(prev_rgb)
    next_padded, _, _ = _pad_to_target(next_rgb)

    device: torch.device = estimator["device"]
    model = estimator["model"]
    transforms = estimator["transforms"]

    prev_t = _to_chw_tensor(prev_padded, device)
    next_t = _to_chw_tensor(next_padded, device)
    # ``transforms`` expects [0, 1] floats and rescales to [-1, 1] plus any
    # other preprocessing the weights were trained with.
    prev_t, next_t = transforms(prev_t, next_t)

    with torch.inference_mode():
        flow_predictions = model(prev_t, next_t, num_flow_updates=_NUM_FLOW_UPDATES)
    # RAFT returns a list of progressive predictions; the last one is final.
    final = flow_predictions[-1]  # (1, 2, Hp, Wp)

    # (1, 2, H, W) -> (H, W, 2) float32 numpy.
    out = final.squeeze(0).permute(1, 2, 0).detach().to("cpu").numpy()
    out = out.astype(np.float32, copy=False)

    # Crop back to original H, W (we padded on bottom/right only).
    out = out[:h, :w, :]
    # Ensure a contiguous float32 array for downstream consumers.
    return np.ascontiguousarray(out)
