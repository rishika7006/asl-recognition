"""Lighting-robustness evaluation using user_videos.

For each (backend, classifier), runs every user clip through the trained
classifier and reports top-1 accuracy split by lighting condition
(``good_light`` vs ``low_light``). Also computes feature similarity
(cosine) between the paired good/low clips of each class — a measure of
how stable the flow features are under a lighting change, independent
of any classifier.

METHODOLOGICAL CAVEAT
=====================
The user_videos clips were also part of the training set, so the
"accuracy" here is inflated by memorization — we expect train-time
recall to be high. What this script reveals is whether the *gap*
between good-light and low-light recall is large (= lighting-sensitive
features) or small (= lighting-robust features). The cosine similarity
of paired feature vectors is a cleaner robustness metric.

Usage:
    python -m src.eval.eval_lighting

Outputs:
    eval_results/lighting_robustness.md
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F

from src.extract_features import (
    _build_name_to_remapped,
    _is_video_file,
    extract_clip_features,
)
from src.flow_backends import get_backend
from src.hand_crop import HandCropper
from src.train import FlowMLP, svm_scores

OUT_DIR = "eval_results"
RESULTS_DIR = "results"
FEATURES_ROOT = "data/features"
USER_VIDEOS_DIR = "data/user_videos"
CONFIG_PATH = "configs/wlasl10.yaml"


def _load_classifier(backend: str, clf_name: str):
    """Return a callable ``X -> (N, C) scores`` for the trained classifier."""
    if clf_name == "SVM":
        clf = joblib.load(os.path.join(RESULTS_DIR, f"svm_{backend}.joblib"))
        return lambda X: svm_scores(clf, X), clf
    blob = torch.load(
        os.path.join(RESULTS_DIR, f"mlp_{backend}.pt"),
        map_location="cpu",
        weights_only=False,
    )
    mlp = FlowMLP(
        input_dim=blob["input_dim"],
        num_classes=blob["num_classes"],
        hidden=tuple(blob["hidden"]),
        dropout=blob["dropout"],
    )
    mlp.load_state_dict(blob["state_dict"])
    mlp.eval()

    def _score(X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.empty((0, 0), dtype=np.float32)
        with torch.no_grad():
            out = F.softmax(mlp(torch.from_numpy(X).float()), dim=1)
        return out.detach().cpu().numpy()

    return _score, mlp


def _extract_user_clip_features(
    backend_name: str, device: str
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract features for every user clip; group by class then by light.

    Returns: ``{class_name: {"good_light": feats, "low_light": feats}}``
    where ``feats`` is the (15, 288) per-clip feature stack from the
    backend (NOT scaled — caller scales).
    """
    import yaml

    cfg = yaml.safe_load(open(CONFIG_PATH))
    ds = cfg["dataset"]
    num_frames = int(ds["num_frames"])
    crop_size = int(ds.get("crop_size", 96))
    grid_size = int(ds.get("grid_size", 12))

    # Need top_classes for the name → remapped index lookup.
    cm_path = os.path.join(FEATURES_ROOT, backend_name, "class_map.json")
    with open(cm_path, "r") as f:
        class_map = json.load(f)
    # Reconstruct top_classes ordered by remapped index 0..K-1
    inv = sorted(class_map.items(), key=lambda kv: int(kv[1]))
    top_classes = [int(k) for k, _ in inv]
    name_to_remapped = _build_name_to_remapped(
        top_classes, "wlasl_class_list.txt"
    )

    backend = get_backend(backend_name)
    estimator = backend.make_estimator(device=device)
    cropper = HandCropper()

    out: Dict[str, Dict[str, np.ndarray]] = {}
    try:
        for entry in sorted(os.listdir(USER_VIDEOS_DIR)):
            class_dir = os.path.join(USER_VIDEOS_DIR, entry)
            if entry.startswith(".") or not os.path.isdir(class_dir):
                continue
            cls_name = entry.lower()
            if cls_name not in name_to_remapped:
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not _is_video_file(fname):
                    continue
                video_path = os.path.join(class_dir, fname)
                cap = cv2.VideoCapture(video_path)
                total = (
                    int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if cap.isOpened()
                    else 0
                )
                cap.release()
                if total <= 1:
                    continue
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
                if feats is None:
                    continue
                # Bucket by light from filename (..._good_light.mov / ..._low_light.mov)
                stem = os.path.splitext(fname)[0].lower()
                if "good_light" in stem:
                    light = "good_light"
                elif "low_light" in stem:
                    light = "low_light"
                else:
                    light = "other"
                out.setdefault(cls_name, {})[light] = feats
    finally:
        cropper.close()
    return out


def _evaluate(
    backend: str,
    clf_name: str,
    feats_by_class_light: Dict[str, Dict[str, np.ndarray]],
    name_to_remapped: Dict[str, int],
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]]:
    """Run classifier on each clip; return per-light correct/total counts.

    Returns:
        good_light_per_class: {class_name: (correct, total)}
        low_light_per_class:  {class_name: (correct, total)}
    """
    scaler = joblib.load(
        os.path.join(FEATURES_ROOT, backend, "scaler.joblib")
    )
    score_fn, _ = _load_classifier(backend, clf_name)

    good: Dict[str, Tuple[int, int]] = {}
    low: Dict[str, Tuple[int, int]] = {}
    for cls_name, light_dict in feats_by_class_light.items():
        true_idx = name_to_remapped[cls_name]
        for light_key, target in (
            ("good_light", good),
            ("low_light", low),
        ):
            feats = light_dict.get(light_key)
            if feats is None:
                target[cls_name] = (0, 0)
                continue
            flat = feats.reshape(1, -1).astype(np.float32)
            scaled = scaler.transform(flat).astype(np.float32)
            scores = score_fn(scaled)
            pred = int(scores.argmax(axis=1)[0])
            target[cls_name] = (
                int(pred == true_idx),
                1,
            )
    return good, low


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _build_name_to_remapped_for_backend(backend: str) -> Dict[str, int]:
    cm_path = os.path.join(FEATURES_ROOT, backend, "class_map.json")
    with open(cm_path, "r") as f:
        class_map = json.load(f)
    inv = sorted(class_map.items(), key=lambda kv: int(kv[1]))
    top_classes = [int(k) for k, _ in inv]
    return _build_name_to_remapped(top_classes, "wlasl_class_list.txt")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    md_lines: List[str] = []
    md_lines.append("# Lighting robustness — good vs low light\n")
    md_lines.append(
        "**Methodological caveat.** The user_videos clips were "
        "included in the training set when the models were retrained, "
        "so the accuracy numbers below are train-time recall (inflated). "
        "What's still informative is the *gap* between good-light and "
        "low-light recall — a small gap means the model handles both "
        "lighting regimes consistently; a large gap means lighting "
        "shifts the feature distribution. The cosine-similarity table "
        "at the bottom is a classifier-independent robustness measure: "
        "how close are the flow features for the same sign filmed "
        "under the two lighting conditions?\n"
    )

    cosine_rows: List[Tuple[str, float, float]] = []  # name, fb_cos, raft_cos

    for backend in ("farneback", "raft"):
        device = "mps" if backend == "raft" and torch.backends.mps.is_available() else "cpu"
        print(f"[{backend}] extracting user-video features…")
        feats_by_cls = _extract_user_clip_features(backend, device=device)
        name_to_remapped = _build_name_to_remapped_for_backend(backend)

        # Cosine sims (one per class).
        for cls_name in sorted(feats_by_cls.keys()):
            good_f = feats_by_cls[cls_name].get("good_light")
            low_f = feats_by_cls[cls_name].get("low_light")
            if good_f is None or low_f is None:
                continue
            sim = _cosine(good_f, low_f)
            # Stash; we'll merge across backends below.
            cosine_rows.append((f"{cls_name}::{backend}", sim, float("nan")))

        for clf_name in ("SVM", "MLP"):
            print(f"[{backend}/{clf_name}] scoring…")
            good, low = _evaluate(
                backend, clf_name, feats_by_cls, name_to_remapped
            )
            md_lines.append(
                f"\n## {backend.capitalize()} — {clf_name}\n"
            )
            md_lines.append(
                "| class | good-light | low-light | gap (good - low) |"
            )
            md_lines.append("|---|---|---|---|")
            n_g_correct = n_g_total = n_l_correct = n_l_total = 0
            for cls_name in sorted(good.keys()):
                g_c, g_t = good[cls_name]
                l_c, l_t = low[cls_name]
                g_acc = (g_c / g_t) if g_t else float("nan")
                l_acc = (l_c / l_t) if l_t else float("nan")
                g_str = "n/a" if g_t == 0 else f"{g_c}/{g_t} ({g_acc:.0%})"
                l_str = "n/a" if l_t == 0 else f"{l_c}/{l_t} ({l_acc:.0%})"
                gap = (
                    "n/a"
                    if (g_t == 0 or l_t == 0)
                    else f"{(g_acc - l_acc):+.0%}"
                )
                md_lines.append(
                    f"| {cls_name} | {g_str} | {l_str} | {gap} |"
                )
                n_g_correct += g_c
                n_g_total += g_t
                n_l_correct += l_c
                n_l_total += l_t
            g_overall = (
                f"{n_g_correct}/{n_g_total} "
                f"({(n_g_correct / max(n_g_total, 1)):.0%})"
            )
            l_overall = (
                f"{n_l_correct}/{n_l_total} "
                f"({(n_l_correct / max(n_l_total, 1)):.0%})"
            )
            gap = (
                f"{(n_g_correct / max(n_g_total, 1)) - (n_l_correct / max(n_l_total, 1)):+.0%}"
            )
            md_lines.append(f"| **TOTAL** | **{g_overall}** | **{l_overall}** | **{gap}** |\n")

    # -------------- cosine sim summary --------------
    # Build a clean per-class dict: {cls_name: {fb: cos, raft: cos}}
    by_cls: Dict[str, Dict[str, float]] = {}
    for key, sim, _ in cosine_rows:
        cls_name, backend = key.split("::")
        by_cls.setdefault(cls_name, {})[backend] = sim
    md_lines.append("\n## Feature similarity (cosine) between good-light and low-light clips of the same sign")
    md_lines.append(
        "\nClassifier-independent. Higher = more lighting-robust feature representation. "
        "Range -1..1; >0.9 is very stable, <0.5 is sensitive.\n"
    )
    md_lines.append("| class | Farnebäck | RAFT |")
    md_lines.append("|---|---|---|")
    for cls_name in sorted(by_cls.keys()):
        fb = by_cls[cls_name].get("farneback", float("nan"))
        rt = by_cls[cls_name].get("raft", float("nan"))
        md_lines.append(f"| {cls_name} | {fb:+.3f} | {rt:+.3f} |")
    fb_vals = [v.get("farneback") for v in by_cls.values() if "farneback" in v]
    rt_vals = [v.get("raft") for v in by_cls.values() if "raft" in v]
    md_lines.append(
        f"| **mean** | **{np.mean(fb_vals):+.3f}** | **{np.mean(rt_vals):+.3f}** |\n"
    )

    out_path = os.path.join(OUT_DIR, "lighting_robustness.md")
    with open(out_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
