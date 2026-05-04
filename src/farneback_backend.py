"""Farnebäck optical-flow backend for the ASL pipeline.

Conforms to the shared backend interface used alongside ``raft_backend``:

    NAME: str
    make_estimator(device: str) -> opaque
    flow(estimator, prev_rgb, next_rgb) -> (H, W, 2) float32

Farnebäck is stateless and runs entirely on CPU through OpenCV, so
``make_estimator`` returns ``None`` and ``flow`` ignores the estimator
argument. Parameters mirror those used by the original
``src/optical_flow.py`` so that the offline-extracted features and the
live demo see the same flow distribution.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

NAME: str = "farneback"

# Farnebäck parameters — identical to the legacy compute_dense_optical_flow
# call in src/optical_flow.py, kept centralized here so changes ripple to
# both training and live inference.
_PYR_SCALE: float = 0.5
_LEVELS: int = 3
_WINSIZE: int = 15
_ITERATIONS: int = 3
_POLY_N: int = 5
_POLY_SIGMA: float = 1.2
_FLAGS: int = 0


def make_estimator(device: str = "cpu") -> Optional[Any]:
    """Return a stateless estimator handle.

    Farnebäck has no learned weights and no device state, so this returns
    ``None``. The ``device`` argument is accepted for API compatibility
    with the RAFT backend but is otherwise ignored.

    Args:
        device: Ignored. Present so callers can swap backends without
            changing call sites.

    Returns:
        ``None``. ``flow`` accepts ``None`` for the estimator argument.
    """
    del device  # unused — Farnebäck is CPU-only and stateless
    return None


def _to_uint8_gray(img: np.ndarray) -> np.ndarray:
    """Convert an (H, W, 3) RGB or (H, W) grayscale array to uint8 grayscale."""
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 2:
        gray = img
    else:
        raise ValueError(
            f"Expected (H, W, 3) RGB or (H, W) grayscale array, got shape {img.shape}"
        )
    if gray.dtype != np.uint8:
        # Float [0, 1] frames are sometimes passed in by mistake; rescale.
        if gray.dtype.kind == "f":
            gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)
    return gray


def flow(
    estimator: Optional[Any],
    prev_rgb: np.ndarray,
    next_rgb: np.ndarray,
) -> np.ndarray:
    """Compute Farnebäck dense optical flow between two RGB frames.

    Args:
        estimator: Ignored (Farnebäck is stateless). Pass the value
            returned by ``make_estimator`` for API consistency.
        prev_rgb: ``(H, W, 3)`` uint8 RGB frame.
        next_rgb: ``(H, W, 3)`` uint8 RGB frame, same H, W as ``prev_rgb``.

    Returns:
        ``(H, W, 2)`` float32 displacement field. Channel 0 is the
        horizontal (u) flow, channel 1 is the vertical (v) flow.
    """
    del estimator  # stateless

    if prev_rgb.shape[:2] != next_rgb.shape[:2]:
        raise ValueError(
            "prev_rgb and next_rgb must share the same H, W; "
            f"got {prev_rgb.shape[:2]} vs {next_rgb.shape[:2]}"
        )

    prev_gray = _to_uint8_gray(prev_rgb)
    next_gray = _to_uint8_gray(next_rgb)

    raw = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        _PYR_SCALE,
        _LEVELS,
        _WINSIZE,
        _ITERATIONS,
        _POLY_N,
        _POLY_SIGMA,
        _FLAGS,
    )
    # OpenCV returns float32 already, but cast defensively to keep contract.
    return raw.astype(np.float32, copy=False)
