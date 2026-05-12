"""Smoke test for the Farnebäck pipeline (HandCropper + extract_features + train).

Run from the project root:

    python3 -m tests.smoke_farneback

Validates that:
    1. All required modules import.
    2. ``flow_backends.get_backend("farneback")`` resolves.
    3. ``HandCropper.bbox_for_clip`` and ``crop_clip`` produce 96x96 crops
       (with a ``None`` fallback for random pixels).
    4. ``features_from_cropped_clip`` returns the expected ``(15, 288)``
       float32 shape for 16 input frames + grid_size=12.
    5. ``grid_pool_flow`` produces the right number of cells.
    6. ``train.run_training`` end-to-end on synthetic features:
       SVM and MLP both fit and save to a temp directory.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import Tuple

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import farneback_backend, flow_backends, hand_crop  # noqa: E402
from src.dataset import _augment_clip, _negate_u_channel  # noqa: E402
from src.extract_features import (  # noqa: E402
    features_from_cropped_clip,
    grid_pool_flow,
)
from src.flow_backends import get_backend  # noqa: E402
from src.hand_crop import HandCropper  # noqa: E402
from src.train import run_training  # noqa: E402


NUM_FRAMES = 16
RAW_SIZE = 256
CROP = 96
GRID = 12
EXPECTED_FEAT_DIM = GRID * GRID * 2
EXPECTED_PAIRS = NUM_FRAMES - 1


def _random_clip(seed: int = 0, size: int = RAW_SIZE, n: int = NUM_FRAMES):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8) for _ in range(n)]


def test_backend_dispatcher() -> None:
    print("[1] backend dispatcher...")
    assert farneback_backend.NAME == "farneback"
    assert "farneback" in flow_backends.available_backends()
    backend = get_backend("farneback")
    assert backend.NAME == "farneback"
    est = backend.make_estimator()
    assert est is None, "Farnebäck estimator should be None (stateless)"
    print("    OK")


def test_grid_pool() -> None:
    print("[2] grid_pool_flow shape...")
    flow = np.zeros((CROP, CROP, 2), dtype=np.float32)
    flow[..., 0] = 1.0
    pooled = grid_pool_flow(flow, GRID)
    assert pooled.shape == (EXPECTED_FEAT_DIM,), pooled.shape
    assert pooled.dtype == np.float32, pooled.dtype
    assert np.allclose(pooled[0::2], 1.0)
    assert np.allclose(pooled[1::2], 0.0)
    print("    OK  pooled.shape =", pooled.shape)


def test_hand_cropper_random() -> None:
    print("[3] HandCropper on random pixels...")
    cropper = HandCropper()
    try:
        frames = _random_clip(seed=7)
        bbox = cropper.bbox_for_clip(frames)
        # On random pixels MediaPipe usually finds no hands, but a stray
        # detection can happen — accept either.
        assert bbox is None or (
            isinstance(bbox, tuple)
            and len(bbox) == 4
            and bbox[0] >= 0
            and bbox[1] >= 0
            and bbox[2] > bbox[0]
            and bbox[3] > bbox[1]
        ), f"unexpected bbox: {bbox}"
        cropped = cropper.crop_clip(frames, bbox, target_size=CROP)
        assert len(cropped) == NUM_FRAMES, len(cropped)
        for f in cropped:
            assert f.shape == (CROP, CROP, 3), f.shape
            assert f.dtype == np.uint8
        bb1 = cropper.bbox_for_frame(frames[0])
        assert bb1 is None or len(bb1) == 4
    finally:
        cropper.close()
    print("    OK  bbox=", bbox, " cropped frames=", len(cropped))


def test_features_from_cropped_clip() -> None:
    print("[4] features_from_cropped_clip output shape...")
    cropper = HandCropper()
    try:
        frames = _random_clip(seed=11)
        bbox = cropper.bbox_for_clip(frames)
        cropped = cropper.crop_clip(frames, bbox, target_size=CROP)
        backend = get_backend("farneback")
        est = backend.make_estimator()
        feats = features_from_cropped_clip(cropped, backend, est, GRID)
        assert feats.shape == (EXPECTED_PAIRS, EXPECTED_FEAT_DIM), feats.shape
        assert feats.dtype == np.float32
        assert np.all(np.isfinite(feats)), "features contain NaN/inf"
    finally:
        cropper.close()
    print("    OK  feats.shape =", feats.shape)


def test_augmentation_helpers() -> None:
    print("[5] augmentation helpers...")
    frames = _random_clip(seed=23)
    aug = _augment_clip(frames, flip=True, angle_deg=3.0, tx=2, ty=-1, brightness=0.1, contrast=1.05)
    assert len(aug) == len(frames)
    assert aug[0].shape == frames[0].shape
    assert aug[0].dtype == np.uint8
    feats = np.stack([np.arange(EXPECTED_FEAT_DIM, dtype=np.float32) for _ in range(EXPECTED_PAIRS)])
    neg = _negate_u_channel(feats, GRID)
    assert np.allclose(neg[:, 0::2], -feats[:, 0::2])
    assert np.allclose(neg[:, 1::2], feats[:, 1::2])
    print("    OK")


def _make_synthetic_features(
    out_dir: str, n_train: int = 30, n_val: int = 9, n_test: int = 9, num_classes: int = 3
) -> Tuple[str, str]:
    """Build a fake features dir with separable per-class signals."""
    rng = np.random.default_rng(42)
    pairs = EXPECTED_PAIRS
    feat = EXPECTED_FEAT_DIM
    os.makedirs(out_dir, exist_ok=True)

    def synth(n: int, seed_offset: int) -> Tuple[np.ndarray, np.ndarray]:
        Xs = np.zeros((n, pairs, feat), dtype=np.float32)
        ys = np.zeros((n,), dtype=np.int64)
        for i in range(n):
            cls = i % num_classes
            base = np.zeros((pairs, feat), dtype=np.float32)
            # Each class gets a constant offset on a different feature index plus mild noise.
            base[:, cls * 5 : (cls + 1) * 5] = 2.5
            noise = rng.standard_normal((pairs, feat)).astype(np.float32) * 0.1
            Xs[i] = base + noise
            ys[i] = cls
        return Xs, ys

    Xtr, ytr = synth(n_train, 0)
    Xva, yva = synth(n_val, 1)
    Xte, yte = synth(n_test, 2)
    np.save(os.path.join(out_dir, "X_train.npy"), Xtr)
    np.save(os.path.join(out_dir, "y_train.npy"), ytr)
    np.save(os.path.join(out_dir, "X_val.npy"), Xva)
    np.save(os.path.join(out_dir, "y_val.npy"), yva)
    np.save(os.path.join(out_dir, "X_test.npy"), Xte)
    np.save(os.path.join(out_dir, "y_test.npy"), yte)

    from sklearn.preprocessing import StandardScaler
    import joblib

    scaler = StandardScaler()
    scaler.fit(Xtr.reshape(Xtr.shape[0], -1))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))

    class_map = {f"class_{i}": i for i in range(num_classes)}
    with open(os.path.join(out_dir, "class_map.json"), "w") as f:
        json.dump(class_map, f)

    return out_dir, "synthetic"


def test_train_end_to_end() -> None:
    print("[6] training end-to-end on synthetic features...")
    tmp = tempfile.mkdtemp(prefix="smoke_farneback_")
    try:
        feat_dir = os.path.join(tmp, "features", "farneback")
        results_dir = os.path.join(tmp, "results")
        _make_synthetic_features(feat_dir)

        cfg = dict(
            dataset=dict(
                json_path="N/A",
                videos_dir="N/A",
                top_k=3,
                num_frames=NUM_FRAMES,
                crop_size=CROP,
                grid_size=GRID,
            ),
            training=dict(
                batch_size=8,
                epochs=20,
                lr=1e-3,
                weight_decay=1e-3,
                early_stop_patience=5,
            ),
        )
        summary = run_training(
            config=cfg,
            backend_name="farneback",
            features_dir=feat_dir,
            results_dir=results_dir,
            device="cpu",
        )
        assert summary["svm"]["test"]["top1"] >= 0.6, summary["svm"]["test"]
        assert summary["mlp"]["test"]["top1"] >= 0.6, summary["mlp"]["test"]
        assert os.path.exists(os.path.join(results_dir, "svm_farneback.joblib"))
        assert os.path.exists(os.path.join(results_dir, "mlp_farneback.pt"))
        assert os.path.exists(os.path.join(results_dir, "scaler_farneback.joblib"))
        assert os.path.exists(os.path.join(results_dir, "class_map.json"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("    OK")


def test_config_yaml_parses() -> None:
    print("[7] configs/wlasl10.yaml parses correctly...")
    with open(os.path.join(ROOT, "configs", "wlasl10.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    ds = cfg["dataset"]
    assert ds["top_k"] == 10
    assert ds["num_classes"] == 10
    assert ds["crop_size"] == 96
    assert ds["grid_size"] == 12
    assert ds["videos_dir"] == "data/raw/videos"
    print("    OK")


def main() -> int:
    test_backend_dispatcher()
    test_grid_pool()
    test_hand_cropper_random()
    test_features_from_cropped_clip()
    test_augmentation_helpers()
    test_config_yaml_parses()
    test_train_end_to_end()
    print("\nOK: smoke_farneback passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
