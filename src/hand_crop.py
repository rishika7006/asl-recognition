"""MediaPipe-based hand bounding-box estimation for the ASL pipeline.

MediaPipe is used to produce a square bounding box around the signer's hands
so the pipeline can crop and zoom in before computing optical flow. Landmark
coordinates are NOT used as features — they're silently zero on low-resolution
WLASL frames where MediaPipe fails to detect, which hurt earlier iterations.

Public API:
    HandCropper.bbox_for_clip(frames)   -> (x1, y1, x2, y2) | None
    HandCropper.bbox_for_frame(frame)   -> (x1, y1, x2, y2) | None
    HandCropper.crop_clip(frames, bbox, target_size) -> list[ndarray]

Uses ``mediapipe.tasks.vision.HandLandmarker`` (mediapipe>=0.10) and auto-
downloads the model file on first use. If the download fails (offline,
sandbox), the cropper returns ``None`` bboxes and ``crop_clip`` falls back
to center cropping so the rest of the pipeline keeps working.
"""

from __future__ import annotations

import logging
import os
import urllib.request
import warnings
from typing import List, Optional, Tuple

import cv2
import numpy as np

_logger = logging.getLogger(__name__)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import mediapipe as mp  # type: ignore[import-untyped]
    from mediapipe.tasks.python import BaseOptions  # type: ignore[import-untyped]
    from mediapipe.tasks.python.vision import (  # type: ignore[import-untyped]
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )


BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) — inclusive-exclusive style

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "mediapipe"
)
_MODEL_PATH = os.path.join(_MODEL_CACHE_DIR, "hand_landmarker.task")


def _ensure_model() -> Optional[str]:
    """Download the hand_landmarker model if not already cached.

    Returns the path on success, or ``None`` if download failed (e.g.
    offline). Callers degrade gracefully when ``None`` is returned.
    """
    if os.path.exists(_MODEL_PATH) and os.path.getsize(_MODEL_PATH) > 0:
        return _MODEL_PATH
    os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)
    try:
        _logger.info("Downloading MediaPipe hand_landmarker model…")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        return _MODEL_PATH
    except Exception as e:  # noqa: BLE001
        _logger.warning(
            "Could not download hand_landmarker model (%s); "
            "HandCropper will return None bboxes (center-crop fallback).",
            e,
        )
        return None


class HandCropper:
    """Find a hand bounding box from MediaPipe and crop frames to it.

    The cropper holds a single MediaPipe HandLandmarker instance. For
    per-clip bbox extraction we run it per frame and take the union of
    all detected landmark positions; this is robust to single-frame
    failures (motion blur, occlusion) as long as at least one frame in
    the clip has a detection.

    The resulting bbox is expanded by ``padding`` (relative to the union
    bbox dimensions) and squared off, so downstream flow features are
    computed on a consistent square crop regardless of hand pose.

    If the underlying MediaPipe model isn't available (download failed,
    sandboxed environment), the cropper still constructs successfully
    but ``bbox_for_*`` always returns ``None``; ``crop_clip`` falls
    back to a center square crop. This keeps the pipeline runnable for
    smoke tests and demos.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        padding: float = 0.30,
    ) -> None:
        """Construct a HandCropper.

        Args:
            max_num_hands: Forwarded to MediaPipe.
            min_detection_confidence: Forwarded to MediaPipe.
            min_tracking_confidence: Forwarded to MediaPipe.
            padding: Fractional bbox expansion. ``0.30`` adds 30% to
                the longer side before re-squaring.
        """
        self._padding = padding
        self._landmarker: Optional[HandLandmarker] = None

        model_path = _ensure_model()
        if model_path is not None:
            try:
                opts = HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=RunningMode.IMAGE,
                    num_hands=max_num_hands,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_hand_presence_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self._landmarker = HandLandmarker.create_from_options(opts)
            except Exception as e:  # noqa: BLE001
                _logger.warning(
                    "Failed to construct HandLandmarker (%s); "
                    "falling back to None bboxes.",
                    e,
                )
                self._landmarker = None

    def _detect_landmarks_xy(
        self, frame_rgb: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Run MediaPipe on one frame; return all (x, y) landmarks in pixel coords."""
        if self._landmarker is None:
            return []
        if frame_rgb.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 RGB frame, got dtype={frame_rgb.dtype}"
            )
        h, w = frame_rgb.shape[:2]
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._landmarker.detect(mp_img)
        except Exception as e:  # noqa: BLE001
            _logger.debug("HandLandmarker.detect failed on a frame: %s", e)
            return []

        coords: List[Tuple[float, float]] = []
        if not result.hand_landmarks:
            return coords
        for hand in result.hand_landmarks:
            for lm in hand:
                coords.append((lm.x * w, lm.y * h))
        return coords

    def _square_bbox_from_points(
        self, points: List[Tuple[float, float]], img_shape: Tuple[int, int]
    ) -> Optional[BBox]:
        """Compute a square, padded, in-bounds bbox enclosing ``points``."""
        if not points:
            return None
        h, w = img_shape
        xs = np.asarray([p[0] for p in points], dtype=np.float32)
        ys = np.asarray([p[1] for p in points], dtype=np.float32)
        x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

        bw = x2 - x1
        bh = y2 - y1
        if bw < 1.0:
            bw = 1.0
        if bh < 1.0:
            bh = 1.0
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        side = max(bw, bh) * (1.0 + self._padding)

        x1 = cx - side / 2.0
        x2 = cx + side / 2.0
        y1 = cy - side / 2.0
        y2 = cy + side / 2.0

        x1 = int(max(0, np.floor(x1)))
        y1 = int(max(0, np.floor(y1)))
        x2 = int(min(w, np.ceil(x2)))
        y2 = int(min(h, np.ceil(y2)))

        cur_w = x2 - x1
        cur_h = y2 - y1
        if cur_w <= 0 or cur_h <= 0:
            return None

        # Re-square after clamping shrinks to the smaller in-bounds side.
        side_after = min(cur_w, cur_h)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = side_after // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, x1 + side_after)
        y2 = min(h, y1 + side_after)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def bbox_for_frame(self, frame: np.ndarray) -> Optional[BBox]:
        """Single-frame bounding box. Used by the live demo loop.

        Args:
            frame: ``(H, W, 3)`` uint8 RGB.

        Returns:
            ``(x1, y1, x2, y2)`` or ``None`` if MediaPipe found no hands.
        """
        pts = self._detect_landmarks_xy(frame)
        return self._square_bbox_from_points(pts, frame.shape[:2])

    def bbox_for_clip(self, frames: List[np.ndarray]) -> Optional[BBox]:
        """Compute the union bbox across an entire clip.

        Runs MediaPipe on every frame and unions all detected landmark
        coordinates. Robust to single-frame failures: we only need
        detections in **at least one** frame.

        Args:
            frames: list of ``(H, W, 3)`` uint8 RGB frames (all same H, W).

        Returns:
            ``(x1, y1, x2, y2)`` covering all detected landmarks across
            the clip, expanded and squared, or ``None`` if no frame had
            a detection.
        """
        if not frames:
            return None
        all_points: List[Tuple[float, float]] = []
        img_shape = frames[0].shape[:2]
        for f in frames:
            if f.shape[:2] != img_shape:
                continue
            all_points.extend(self._detect_landmarks_xy(f))
        return self._square_bbox_from_points(all_points, img_shape)

    @staticmethod
    def _center_square_crop(frame: np.ndarray) -> np.ndarray:
        """Return a centered square crop of ``frame`` (no resize)."""
        h, w = frame.shape[:2]
        side = min(h, w)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        return frame[y1 : y1 + side, x1 : x1 + side]

    def crop_clip(
        self,
        frames: List[np.ndarray],
        bbox: Optional[BBox],
        target_size: int,
    ) -> List[np.ndarray]:
        """Crop each frame to ``bbox`` and resize to ``target_size`` square.

        Args:
            frames: list of ``(H, W, 3)`` uint8 RGB frames.
            bbox: ``(x1, y1, x2, y2)`` from ``bbox_for_clip`` /
                ``bbox_for_frame``, or ``None`` to fall back to a center
                square crop of each frame.
            target_size: side length (px) of the resized output.

        Returns:
            List of ``(target_size, target_size, 3)`` uint8 RGB frames.
        """
        if not frames:
            return []
        out: List[np.ndarray] = []
        if bbox is None:
            _logger.warning(
                "HandCropper.crop_clip called with bbox=None; "
                "falling back to center square crop."
            )
        for f in frames:
            if bbox is None:
                cropped = self._center_square_crop(f)
            else:
                x1, y1, x2, y2 = bbox
                cropped = f[y1:y2, x1:x2]
                if cropped.size == 0:
                    cropped = self._center_square_crop(f)
            resized = cv2.resize(
                cropped, (target_size, target_size), interpolation=cv2.INTER_AREA
            )
            out.append(resized)
        return out

    def close(self) -> None:
        """Release MediaPipe resources. Safe to call multiple times."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # noqa: BLE001
                pass
            self._landmarker = None

    def __enter__(self) -> "HandCropper":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401, ANN001
        self.close()

    def __del__(self) -> None:  # noqa: D401
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
