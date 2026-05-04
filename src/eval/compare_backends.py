"""Side-by-side comparison of Farnebäck and RAFT pipelines.

Loads the four trained classifiers (FB-SVM, FB-MLP, RAFT-SVM, RAFT-MLP),
re-evaluates each on the train/val/test splits of its corresponding
backend, and emits a single comparison table + per-class report +
confusion-matrix figure into ``eval_results/``.

Usage:
    python -m src.eval.compare_backends

Outputs:
    eval_results/comparison_table.md
    eval_results/comparison_table.csv
    eval_results/per_class_metrics.md
    eval_results/confusion_matrices.png
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.train import FlowMLP, svm_scores

OUT_DIR = "eval_results"
RESULTS_DIR = "results"
FEATURES_ROOT = "data/features"
CLASS_LIST = "wlasl_class_list.txt"


# ----------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------

def _resolve_class_names(backend: str) -> List[str]:
    """Build [name_for_idx_0, name_for_idx_1, ...] using class_map +
    wlasl_class_list.txt."""
    cm_path = os.path.join(FEATURES_ROOT, backend, "class_map.json")
    with open(cm_path, "r") as f:
        class_map = json.load(f)  # {orig_id_str: remapped_idx}
    text_map: Dict[int, str] = {}
    if os.path.exists(CLASS_LIST):
        with open(CLASS_LIST, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    try:
                        text_map[int(parts[0])] = parts[1]
                    except ValueError:
                        pass
    inv = sorted(class_map.items(), key=lambda kv: int(kv[1]))
    out: List[str] = []
    for orig_id_str, _ in inv:
        try:
            out.append(text_map.get(int(orig_id_str), orig_id_str))
        except ValueError:
            out.append(orig_id_str)
    return out


def _load_split_features(backend: str) -> Dict[str, np.ndarray]:
    fdir = os.path.join(FEATURES_ROOT, backend)
    return {
        "X_train": np.load(os.path.join(fdir, "X_train.npy")),
        "y_train": np.load(os.path.join(fdir, "y_train.npy")),
        "X_val": np.load(os.path.join(fdir, "X_val.npy")),
        "y_val": np.load(os.path.join(fdir, "y_val.npy")),
        "X_test": np.load(os.path.join(fdir, "X_test.npy")),
        "y_test": np.load(os.path.join(fdir, "y_test.npy")),
    }


def _flatten_and_scale(X: np.ndarray, scaler) -> np.ndarray:
    flat = X.reshape(X.shape[0], -1) if X.ndim == 3 else X
    return scaler.transform(flat.astype(np.float32)).astype(np.float32)


def _load_mlp(backend: str) -> Tuple[FlowMLP, dict]:
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
    return mlp, blob


def _mlp_scores(mlp: FlowMLP, X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)
    with torch.no_grad():
        out = mlp(torch.from_numpy(X).float())
        out = torch.softmax(out, dim=1)
    return out.detach().cpu().numpy().astype(np.float32)


def _topk_acc(scores: np.ndarray, y_true: np.ndarray, k: int) -> float:
    if scores.size == 0:
        return float("nan")
    k = min(k, scores.shape[1])
    topk = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([yt in row for yt, row in zip(y_true, topk)]))


def _measure_ms_per_clip(fn, X: np.ndarray, n_warmup: int = 5) -> float:
    if X.shape[0] == 0:
        return float("nan")
    n = X.shape[0]
    for i in range(min(n_warmup, n)):
        fn(X[i : i + 1])
    t0 = time.perf_counter()
    for i in range(n):
        fn(X[i : i + 1])
    return 1000.0 * (time.perf_counter() - t0) / n


# ----------------------------------------------------------------------
# evaluation per (backend, classifier)
# ----------------------------------------------------------------------

def _eval_one(
    backend: str, classifier_name: str
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray, List[str], float]:
    """Return ((per_split_metrics, confusion_matrix, class_names, ms/clip)).

    per_split_metrics[split] = {top1, top3, f1_macro}
    """
    feat = _load_split_features(backend)
    scaler = joblib.load(
        os.path.join(FEATURES_ROOT, backend, "scaler.joblib")
    )
    Xtr = _flatten_and_scale(feat["X_train"], scaler)
    Xva = _flatten_and_scale(feat["X_val"], scaler)
    Xte = _flatten_and_scale(feat["X_test"], scaler)

    if classifier_name == "SVM":
        svm = joblib.load(
            os.path.join(RESULTS_DIR, f"svm_{backend}.joblib")
        )
        scores_tr = svm_scores(svm, Xtr)
        scores_va = svm_scores(svm, Xva)
        scores_te = svm_scores(svm, Xte)
        ms_per_clip = _measure_ms_per_clip(
            lambda x: svm_scores(svm, x), Xte
        )
    elif classifier_name == "MLP":
        mlp, _blob = _load_mlp(backend)
        scores_tr = _mlp_scores(mlp, Xtr)
        scores_va = _mlp_scores(mlp, Xva)
        scores_te = _mlp_scores(mlp, Xte)
        ms_per_clip = _measure_ms_per_clip(lambda x: _mlp_scores(mlp, x), Xte)
    else:
        raise ValueError(f"Unknown classifier {classifier_name}")

    per_split: Dict[str, Dict[str, float]] = {}
    for split, scores, y in (
        ("train", scores_tr, feat["y_train"]),
        ("val", scores_va, feat["y_val"]),
        ("test", scores_te, feat["y_test"]),
    ):
        if scores.size == 0:
            per_split[split] = {"top1": float("nan"), "top3": float("nan"), "f1": float("nan")}
            continue
        y_pred = scores.argmax(axis=1)
        per_split[split] = {
            "top1": float(accuracy_score(y, y_pred)),
            "top3": _topk_acc(scores, y, 3),
            "f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        }

    # Confusion matrix on test (with explicit labels so missing classes
    # show as zero rows rather than getting collapsed).
    class_names = _resolve_class_names(backend)
    labels = list(range(len(class_names)))
    if scores_te.size:
        cm = confusion_matrix(
            feat["y_test"], scores_te.argmax(axis=1), labels=labels
        )
    else:
        cm = np.zeros((len(labels), len(labels)), dtype=np.int64)

    return per_split, cm, class_names, ms_per_clip


def _per_class_text_report(
    backend: str, classifier_name: str, class_names: List[str]
) -> str:
    feat = _load_split_features(backend)
    scaler = joblib.load(
        os.path.join(FEATURES_ROOT, backend, "scaler.joblib")
    )
    Xte = _flatten_and_scale(feat["X_test"], scaler)
    if classifier_name == "SVM":
        clf = joblib.load(os.path.join(RESULTS_DIR, f"svm_{backend}.joblib"))
        scores = svm_scores(clf, Xte)
    else:
        mlp, _ = _load_mlp(backend)
        scores = _mlp_scores(mlp, Xte)
    if scores.size == 0:
        return "(empty test split)"
    y_pred = scores.argmax(axis=1)
    return classification_report(
        feat["y_test"],
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )


# ----------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------

def _render_table_md(rows: List[Dict[str, str]]) -> str:
    headers = [
        "Backend",
        "Classifier",
        "train top-1",
        "val top-1",
        "test top-1",
        "test top-3",
        "test F1 (macro)",
        "ms/clip (test)",
    ]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append(
            "| " + " | ".join(str(r[h]) for h in headers) + " |"
        )
    return "\n".join(lines)


def _render_confusion_grid(
    confusions: Dict[str, np.ndarray],
    class_names: List[str],
    out_path: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_order = [
        ("Farnebäck — SVM", confusions["farneback_SVM"]),
        ("Farnebäck — MLP", confusions["farneback_MLP"]),
        ("RAFT — SVM", confusions["raft_SVM"]),
        ("RAFT — MLP", confusions["raft_MLP"]),
    ]
    for ax, (title, cm) in zip(axes.flat, plot_order):
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm[i, j]
                if v > 0:
                    ax.text(
                        j,
                        i,
                        str(v),
                        ha="center",
                        va="center",
                        color="white" if v > cm.max() / 2 else "black",
                        fontsize=8,
                    )
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(
        "Test-set confusion matrices (rows = true class, columns = predicted)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows: List[Dict[str, str]] = []
    confusions: Dict[str, np.ndarray] = {}
    per_class_blocks: List[str] = []
    class_names_for_grid: List[str] = []

    for backend in ("farneback", "raft"):
        for clf_name in ("SVM", "MLP"):
            metrics, cm, class_names, ms_per = _eval_one(backend, clf_name)
            confusions[f"{backend}_{clf_name}"] = cm
            class_names_for_grid = class_names

            rows.append(
                {
                    "Backend": backend.capitalize(),
                    "Classifier": clf_name,
                    "train top-1": f"{metrics['train']['top1']:.3f}",
                    "val top-1": f"{metrics['val']['top1']:.3f}",
                    "test top-1": f"{metrics['test']['top1']:.3f}",
                    "test top-3": f"{metrics['test']['top3']:.3f}",
                    "test F1 (macro)": f"{metrics['test']['f1']:.3f}",
                    "ms/clip (test)": f"{ms_per:.2f}",
                }
            )

            block = (
                f"### {backend.capitalize()} — {clf_name}\n\n```\n"
                + _per_class_text_report(backend, clf_name, class_names)
                + "\n```\n"
            )
            per_class_blocks.append(block)

    # comparison_table.md
    md = (
        "# Backend × classifier comparison\n\n"
        "All four models were trained on the same WLASL-10 train split "
        "(plus 20 user_videos clips, 2 per class) and evaluated on the "
        "WLASL-10 test split (13 clips). Latencies measured on CPU.\n\n"
        + _render_table_md(rows)
        + "\n"
    )
    with open(os.path.join(OUT_DIR, "comparison_table.md"), "w") as f:
        f.write(md)

    # comparison_table.csv
    headers = list(rows[0].keys())
    with open(os.path.join(OUT_DIR, "comparison_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # per_class_metrics.md
    with open(os.path.join(OUT_DIR, "per_class_metrics.md"), "w") as f:
        f.write(
            "# Per-class precision / recall / F1 on the WLASL-10 test split\n\n"
            "Note: the test split has only 13 samples spread across "
            "8 of 10 classes; classes with `support=0` had no test "
            "examples and are reported as zeros for completeness.\n\n"
        )
        f.write("\n".join(per_class_blocks))

    # confusion_matrices.png
    _render_confusion_grid(
        confusions,
        class_names_for_grid,
        os.path.join(OUT_DIR, "confusion_matrices.png"),
    )

    print(f"Wrote {OUT_DIR}/comparison_table.md")
    print(f"Wrote {OUT_DIR}/comparison_table.csv")
    print(f"Wrote {OUT_DIR}/per_class_metrics.md")
    print(f"Wrote {OUT_DIR}/confusion_matrices.png")
    print()
    print("Final comparison table:")
    print(_render_table_md(rows))


if __name__ == "__main__":
    main()
