"""Feature extraction for the ASL pipeline (Farnebäck or RAFT).

Pipeline per video:

    1. Read raw .mp4, get total frame count.
    2. Use the action window [start, end] from nslt_100.json to limit
       sampling to the actual sign (avoids picking up frames before/after).
    3. Sample ``num_frames`` (default 16) evenly-spaced frames in the window.
    4. Run MediaPipe Hands on those frames; take the union bbox across
       frames, expand 30%, square it, clamp to bounds. Crop+resize to 96x96.
    5. For each consecutive pair, compute optical flow with the chosen backend.
    6. Pool flow with a 12x12 grid mean → 144 cells x 2 channels = 288
       features per pair → ``(15, 288)`` per clip.
    7. Stack by subset, write npy files plus a StandardScaler fit on train.

MediaPipe's only role is producing the bbox — landmarks are not used
as features.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.flow_backends import get_backend
from src.hand_crop import HandCropper

_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_config(path: str) -> Dict:
    """Load a YAML config file from ``path``."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _read_frames_at(
    cap: cv2.VideoCapture, indices: List[int]
) -> List[np.ndarray]:
    """Read the listed frames from an OpenCV VideoCapture as RGB uint8.

    Returns frames in the same order as ``indices``. Frames that cannot
    be read are silently skipped (so callers should check the length).
    """
    out: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        out.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return out


def _sample_indices_in_window(
    total_frames: int,
    start: int,
    end: int,
    num_frames: int,
) -> List[int]:
    """Pick ``num_frames`` evenly-spaced frame indices inside [start, end].

    Clamps the window to ``[0, total_frames - 1]``. If the window is
    empty/invalid, falls back to evenly sampling the full video.
    """
    if total_frames <= 0:
        return []
    last = total_frames - 1
    s = max(0, min(int(start), last))
    e = max(0, min(int(end), last))
    if e <= s:
        s, e = 0, last
    if e == s:
        return [s] * num_frames
    return list(np.linspace(s, e, num_frames, dtype=int))


def grid_pool_flow(flow: np.ndarray, grid_size: int) -> np.ndarray:
    """Mean-pool a flow field over a ``grid_size`` x ``grid_size`` grid.

    Args:
        flow: ``(H, W, 2)`` float32 displacement field.
        grid_size: number of cells per side.

    Returns:
        ``(grid_size * grid_size * 2,)`` float32 vector — for each cell,
        ``[mean_u, mean_v]`` concatenated in row-major order.
    """
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"flow must be (H, W, 2), got {flow.shape}")
    h, w, _ = flow.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    if cell_h == 0 or cell_w == 0:
        raise ValueError(
            f"grid_size={grid_size} too large for flow shape {flow.shape}"
        )
    out = np.empty(grid_size * grid_size * 2, dtype=np.float32)
    k = 0
    for r in range(grid_size):
        for c in range(grid_size):
            cell = flow[
                r * cell_h : (r + 1) * cell_h,
                c * cell_w : (c + 1) * cell_w,
            ]
            out[k] = float(cell[..., 0].mean())
            out[k + 1] = float(cell[..., 1].mean())
            k += 2
    return out


def features_from_cropped_clip(
    cropped_frames: List[np.ndarray],
    backend,
    estimator,
    grid_size: int,
) -> np.ndarray:
    """Compute the ``(num_pairs, grid_size**2 * 2)`` feature stack.

    Args:
        cropped_frames: list of ``(H, W, 3)`` uint8 RGB frames (already
            cropped to the hand bbox and resized to a fixed square).
        backend: backend module (with ``flow`` callable) from
            ``src.flow_backends.get_backend``.
        estimator: opaque estimator from ``backend.make_estimator``
            (may be ``None`` for stateless backends like Farnebäck).
        grid_size: spatial pooling grid size.

    Returns:
        ``(T - 1, grid_size**2 * 2)`` float32 array.
    """
    if len(cropped_frames) < 2:
        raise ValueError(
            f"Need at least 2 frames to compute flow, got {len(cropped_frames)}"
        )
    feats = []
    for i in range(1, len(cropped_frames)):
        f = backend.flow(estimator, cropped_frames[i - 1], cropped_frames[i])
        feats.append(grid_pool_flow(f, grid_size))
    return np.stack(feats, axis=0).astype(np.float32, copy=False)


def extract_clip_features(
    video_path: str,
    action_window: Tuple[int, int],
    num_frames: int,
    crop_size: int,
    grid_size: int,
    cropper: HandCropper,
    backend,
    estimator,
) -> Optional[np.ndarray]:
    """End-to-end feature extraction for one video file.

    Returns:
        ``(num_frames - 1, grid_size**2 * 2)`` float32 array, or
        ``None`` if the video could not be processed.
    """
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 1:
            return None
        start, end = action_window
        indices = _sample_indices_in_window(total, start, end, num_frames)
        if len(indices) < 2:
            return None
        frames = _read_frames_at(cap, indices)
    finally:
        cap.release()
    if len(frames) < 2:
        return None
    # Pad with the last frame so downstream sees a consistent T = num_frames.
    while len(frames) < num_frames:
        frames.append(frames[-1])

    bbox = cropper.bbox_for_clip(frames)
    cropped = cropper.crop_clip(frames, bbox, target_size=crop_size)
    if len(cropped) < 2:
        return None
    return features_from_cropped_clip(cropped, backend, estimator, grid_size)


def _select_top_k_classes(data: Dict, top_k: int) -> List[int]:
    """Return the ``top_k`` most-populated class ids in ``nslt_100.json``."""
    counts = Counter(info["action"][0] for info in data.values())
    return [c for c, _ in counts.most_common(top_k)]


def _load_missing_set(path: Optional[str]) -> set[str]:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def _build_name_to_remapped(
    top_classes: List[int], class_list_path: str
) -> Dict[str, int]:
    """Build a {gloss_name: remapped_index} dict for the trained class set.

    Reads ``wlasl_class_list.txt`` (tab-separated ``orig_id\tgloss``)
    and returns only the entries whose ``orig_id`` is in ``top_classes``,
    mapping each gloss name to its 0..K-1 remapped index.
    """
    name_to_orig: Dict[str, int] = {}
    if os.path.exists(class_list_path):
        with open(class_list_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    try:
                        name_to_orig[parts[1].lower()] = int(parts[0])
                    except ValueError:
                        pass
    out: Dict[str, int] = {}
    for name, orig in name_to_orig.items():
        if orig in top_classes:
            out[name] = top_classes.index(orig)
    return out


def _is_video_file(fname: str) -> bool:
    """True if ``fname`` looks like a video file we can read with OpenCV."""
    if fname.startswith("."):
        return False
    return os.path.splitext(fname)[1].lower() in {
        ".mp4", ".mov", ".avi", ".mkv", ".m4v"
    }


def _ingest_user_videos(
    user_dir: str,
    name_to_remapped: Dict[str, int],
    num_frames: int,
    crop_size: int,
    grid_size: int,
    cropper: HandCropper,
    backend,
    estimator,
):
    """Yield ``(feats, class_idx, source_path)`` for each user clip.

    Expected layout:
        ``<user_dir>/<class_name>/<clip>.{mov,mp4,...}``

    The whole clip is treated as the action window (no nslt_100 metadata).
    Class folders whose name doesn't match a trained gloss are skipped
    with a warning.
    """
    if not os.path.isdir(user_dir):
        return
    for entry in sorted(os.listdir(user_dir)):
        class_dir = os.path.join(user_dir, entry)
        if entry.startswith(".") or not os.path.isdir(class_dir):
            continue
        cls_name = entry.lower()
        if cls_name not in name_to_remapped:
            _logger.warning(
                "user_videos: skipping folder %r — not in trained class "
                "set %s",
                cls_name,
                sorted(name_to_remapped.keys()),
            )
            continue
        cls_idx = name_to_remapped[cls_name]
        for fname in sorted(os.listdir(class_dir)):
            if not _is_video_file(fname):
                continue
            video_path = os.path.join(class_dir, fname)
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            cap.release()
            if total <= 1:
                _logger.warning(
                    "user_videos: %s has no readable frames", video_path
                )
                continue
            try:
                feats = extract_clip_features(
                    video_path=video_path,
                    action_window=(0, total - 1),
                    num_frames=num_frames,
                    crop_size=crop_size,
                    grid_size=grid_size,
                    cropper=cropper,
                    backend=backend,
                    estimator=estimator,
                )
            except Exception as e:  # noqa: BLE001
                _logger.warning(
                    "user_videos: skipping %s: %s", video_path, e
                )
                feats = None
            if feats is None:
                continue
            yield feats, cls_idx, video_path


def run_extraction(
    config: Dict,
    backend_name: str,
    output_dir: Optional[str] = None,
    device: str = "cpu",
) -> str:
    """Run the full extraction pipeline.

    Args:
        config: parsed YAML config dict.
        backend_name: ``"farneback"`` or ``"raft"``.
        output_dir: where to write features. Defaults to
            ``data/features/{backend_name}``.
        device: device for backends that care (RAFT). Ignored by Farnebäck.

    Returns:
        Path to the output directory.
    """
    ds = config["dataset"]
    json_path = ds["json_path"]
    videos_dir = ds["videos_dir"]
    missing_txt = ds.get("missing_txt")
    top_k = int(ds["top_k"])
    num_frames = int(ds["num_frames"])
    crop_size = int(ds.get("crop_size", 96))
    grid_size = int(ds.get("grid_size", 12))

    if output_dir is None:
        output_dir = os.path.join("data", "features", backend_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    top_classes = _select_top_k_classes(data, top_k)
    class_map = {orig: idx for idx, orig in enumerate(top_classes)}
    missing = _load_missing_set(missing_txt)

    video_list = [
        (vid, info)
        for vid, info in data.items()
        if vid not in missing and info["action"][0] in top_classes
    ]
    _logger.info(
        "Extracting %s features for %d videos (top_k=%d, crop=%d, grid=%d)",
        backend_name,
        len(video_list),
        top_k,
        crop_size,
        grid_size,
    )

    # RAFT loads its model here.
    backend = get_backend(backend_name)
    estimator = backend.make_estimator(device=device)

    cropper = HandCropper()

    X_dict: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    y_dict: Dict[str, List[int]] = {"train": [], "val": [], "test": []}

    n_skipped = 0
    n_user = 0
    try:
        for vid, info in tqdm(video_list, desc=f"{backend_name} extract"):
            subset = info["subset"]
            if subset not in X_dict:
                continue
            class_id = class_map[info["action"][0]]
            start = int(info["action"][1])
            end = int(info["action"][2])

            video_path = os.path.join(videos_dir, f"{vid}.mp4")
            try:
                feats = extract_clip_features(
                    video_path=video_path,
                    action_window=(start, end),
                    num_frames=num_frames,
                    crop_size=crop_size,
                    grid_size=grid_size,
                    cropper=cropper,
                    backend=backend,
                    estimator=estimator,
                )
            except Exception as e:  # noqa: BLE001
                _logger.warning("skipping %s: %s", vid, e)
                feats = None
            if feats is None:
                n_skipped += 1
                continue
            X_dict[subset].append(feats)
            y_dict[subset].append(class_id)

        # User-recorded clips at data/user_videos/<class>/<clip> all go into
        # the TRAIN split — 1-2 clips per class is too few for val/test.
        user_dir = ds.get("user_videos_dir") or os.path.join(
            "data", "user_videos"
        )
        if os.path.isdir(user_dir):
            name_to_remapped = _build_name_to_remapped(
                top_classes, "wlasl_class_list.txt"
            )
            for feats, cls_idx, src in _ingest_user_videos(
                user_dir,
                name_to_remapped,
                num_frames,
                crop_size,
                grid_size,
                cropper,
                backend,
                estimator,
            ):
                X_dict["train"].append(feats)
                y_dict["train"].append(cls_idx)
                _logger.info(
                    "user_videos: + train cls=%d %s", cls_idx, src
                )
                n_user += 1
            _logger.info(
                "Added %d user-recorded clips from %s to TRAIN",
                n_user,
                user_dir,
            )
    finally:
        cropper.close()

    _logger.info(
        "Skipped %d / %d WLASL videos (missing files, read errors, etc.)",
        n_skipped,
        len(video_list),
    )

    feat_per_pair = (grid_size * grid_size) * 2
    pairs = num_frames - 1
    for subset in ("train", "val", "test"):
        if X_dict[subset]:
            X_arr = np.stack(X_dict[subset], axis=0).astype(np.float32)
            y_arr = np.asarray(y_dict[subset], dtype=np.int64)
        else:
            X_arr = np.empty((0, pairs, feat_per_pair), dtype=np.float32)
            y_arr = np.empty((0,), dtype=np.int64)
        np.save(os.path.join(output_dir, f"X_{subset}.npy"), X_arr)
        np.save(os.path.join(output_dir, f"y_{subset}.npy"), y_arr)
        _logger.info("  %-5s X=%s y=%s", subset, X_arr.shape, y_arr.shape)

    if X_dict["train"]:
        X_train = np.stack(X_dict["train"], axis=0).astype(np.float32)
        scaler = StandardScaler()
        scaler.fit(X_train.reshape(X_train.shape[0], -1))
        joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
        _logger.info("  scaler saved (n_train=%d)", X_train.shape[0])
    else:
        _logger.warning("No training samples — scaler not fit.")

    with open(os.path.join(output_dir, "class_map.json"), "w") as f:
        json.dump({str(k): v for k, v in class_map.items()}, f, indent=2)

    return output_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract pooled optical-flow features for WLASL clips."
    )
    parser.add_argument(
        "--backend",
        choices=("farneback", "raft"),
        required=True,
        help="Optical-flow backend.",
    )
    parser.add_argument(
        "--config",
        default="configs/wlasl10.yaml",
        help="YAML config (defaults to configs/wlasl10.yaml).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for RAFT (cpu / cuda / mps). Ignored for Farnebäck.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output dir (default: data/features/{backend}).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out = run_extraction(
        config=cfg,
        backend_name=args.backend,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(f"Features written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
