"""Train an RBF SVM and a small MLP on pre-extracted flow features.

This script replaces the legacy ``src/model.py`` four-classifier sklearn
bake-off. Per the proposal we keep one SVM and one lightweight neural
network — both proposal-faithful — and report top-1, top-3, macro F1,
per-class precision/recall/F1, confusion matrix, and per-clip
inference time.

Inputs (produced by ``src.extract_features``):
    data/features/{backend}/X_{train,val,test}.npy
    data/features/{backend}/y_{train,val,test}.npy
    data/features/{backend}/scaler.joblib
    data/features/{backend}/class_map.json

Outputs:
    results/svm_{backend}.joblib
    results/mlp_{backend}.pt
    results/scaler_{backend}.joblib  (copy of the extractor's scaler)
    results/class_map.json           (copy of the extractor's class map)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import SVC

_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


# ----------------------------------------------------------------------
# config / IO
# ----------------------------------------------------------------------


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _features_dir(backend: str, override: Optional[str]) -> str:
    if override:
        return override
    return os.path.join("data", "features", backend)


def load_features(features_dir: str) -> Dict[str, np.ndarray]:
    """Load X/y per split plus scaler and class_map."""
    out: Dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        out[f"X_{split}"] = np.load(os.path.join(features_dir, f"X_{split}.npy"))
        out[f"y_{split}"] = np.load(os.path.join(features_dir, f"y_{split}.npy"))
    out["scaler"] = joblib.load(os.path.join(features_dir, "scaler.joblib"))
    with open(os.path.join(features_dir, "class_map.json"), "r") as f:
        out["class_map"] = json.load(f)
    return out


def _flatten(X: np.ndarray) -> np.ndarray:
    """Reshape (N, T-1, F) → (N, (T-1)*F)."""
    return X.reshape(X.shape[0], -1) if X.ndim == 3 else X


def _scale(scaler, X: np.ndarray) -> np.ndarray:
    flat = _flatten(X).astype(np.float32, copy=False)
    return scaler.transform(flat).astype(np.float32, copy=False)


# ----------------------------------------------------------------------
# evaluation utilities
# ----------------------------------------------------------------------


def _topk_accuracy(scores: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """Return top-k accuracy from class scores ``(N, C)``."""
    if scores.size == 0:
        return float("nan")
    k = min(k, scores.shape[1])
    topk = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([y in row for y, row in zip(y_true, topk)]))


def _format_section(title: str) -> str:
    bar = "=" * 70
    return f"\n{bar}\n{title}\n{bar}"


def evaluate_split(
    name: str,
    classifier_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Print and return common classification metrics for one split."""
    if scores.size == 0:
        _logger.info("[%s] %s: empty split, skipping", classifier_name, name)
        return dict(top1=float("nan"), top3=float("nan"), f1_macro=float("nan"))
    y_pred = scores.argmax(axis=1)
    top1 = accuracy_score(y_true, y_pred)
    top3 = _topk_accuracy(scores, y_true, k=3)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n[{classifier_name}] {name}: top1={top1:.4f} top3={top3:.4f} f1_macro={f1m:.4f}")
    return dict(top1=float(top1), top3=float(top3), f1_macro=float(f1m))


def detailed_report(
    name: str,
    classifier_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> None:
    """Print sklearn classification_report + confusion matrix for ``name``.

    Passes an explicit ``labels=[0..K-1]`` so the report includes rows for
    classes that happen to be absent from ``y_true`` (common with our
    13-sample test set covering only 8 of 10 classes).
    """
    if y_true.size == 0:
        return
    print(_format_section(f"{classifier_name} — {name} detailed report"))
    labels = list(range(len(target_names))) if target_names else None
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
            digits=4,
        )
    )
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=labels))


# ----------------------------------------------------------------------
# SVM
# ----------------------------------------------------------------------


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    gamma: str = "scale",
    random_state: int = 42,
) -> SVC:
    """Train an RBF SVM.

    ``probability=True`` is intentionally NOT set: with only ~8 samples
    per class, the internal 5-fold Platt-scaling cross-validation can't
    fit reliable sigmoids and ``predict_proba`` ends up returning near-
    uniform noise whose argmax disagrees with ``predict()``. We score
    via ``decision_function`` + softmax instead — see :func:`svm_scores`.
    """
    svm = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        random_state=random_state,
    )
    svm.fit(X_train, y_train)
    return svm


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def svm_scores(svm: SVC, X: np.ndarray) -> np.ndarray:
    """Return ``(N, C)`` probability-like scores from a fitted SVM.

    Uses ``decision_function`` (consistent with ``predict()``) and a
    softmax to produce a probability-like vector for top-k ranking and
    confidence display. Matches the input-shape contract of
    ``predict_proba`` so downstream code is unchanged.
    """
    if X.shape[0] == 0:
        n_classes = len(svm.classes_) if hasattr(svm, "classes_") else 0
        return np.empty((0, n_classes), dtype=np.float32)
    raw = svm.decision_function(X)
    if raw.ndim == 1:
        # binary case
        raw = np.column_stack([-raw, raw])
    return _softmax_rows(raw).astype(np.float32, copy=False)


# ----------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------


class FlowMLP(nn.Module):
    """Small MLP over flattened pooled-flow features.

    Architecture: two hidden layers with ReLU + dropout. Default hidden
    sizes ``(256, 128)`` give roughly 80–200k params depending on the
    input dimension (e.g. 15*288=4320 → 256 → 128 → 10 ≈ 1.14M
    params; the smaller (128, 64) head is used when inputs are very
    high dim).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden: Tuple[int, int] = (256, 128),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    early_stop_patience: int = 10,
    hidden: Tuple[int, int] = (256, 128),
    dropout: float = 0.5,
    device: Optional[str] = None,
    seed: int = 42,
) -> Tuple[FlowMLP, Dict[str, float]]:
    """Train ``FlowMLP`` with Adam + early stopping on val accuracy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model = FlowMLP(
        input_dim=X_train.shape[1], num_classes=num_classes, hidden=hidden, dropout=dropout
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt = torch.from_numpy(X_train).float().to(dev)
    yt = torch.from_numpy(y_train).long().to(dev)
    Xv = torch.from_numpy(X_val).float().to(dev)
    yv = torch.from_numpy(y_val).long().to(dev)

    best_val = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    no_improve = 0

    n = Xt.shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = F.cross_entropy(out, yt[idx])
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * idx.shape[0]

        model.eval()
        with torch.no_grad():
            if Xv.shape[0] == 0:
                val_acc = float("nan")
            else:
                val_pred = model(Xv).argmax(dim=1)
                val_acc = float((val_pred == yv).float().mean().item())

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch % 10 == 0 or epoch == 1:
            _logger.info(
                "MLP epoch %3d: train_loss=%.4f  val_top1=%.4f  best=%.4f",
                epoch,
                total_loss / max(n, 1),
                val_acc,
                best_val,
            )
        if no_improve >= early_stop_patience:
            _logger.info("Early stop at epoch %d (no improve %d)", epoch, no_improve)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, dict(best_val_top1=float(best_val))


def mlp_scores(model: FlowMLP, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Run the MLP and return softmax scores ``(N, C)`` on CPU."""
    model.eval()
    if X.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)
    with torch.no_grad():
        out = model(torch.from_numpy(X).float().to(device))
        out = F.softmax(out, dim=1)
    return out.detach().cpu().numpy()


# ----------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------


def _measure_inference_time(
    fn, X: np.ndarray, n_warmup: int = 5
) -> float:
    """Mean per-clip inference time (ms) for a callable ``fn(X[i:i+1])``."""
    if X.shape[0] == 0:
        return float("nan")
    n = X.shape[0]
    n_warmup = min(n_warmup, n)
    for i in range(n_warmup):
        fn(X[i : i + 1])
    t0 = time.perf_counter()
    for i in range(n):
        fn(X[i : i + 1])
    elapsed = time.perf_counter() - t0
    return 1000.0 * elapsed / n


def _target_names_from_class_map(class_map: Dict[str, int]) -> List[str]:
    """Build a ``[class_name]`` list ordered by remapped index."""
    pairs = sorted(class_map.items(), key=lambda kv: kv[1])
    return [str(k) for k, _ in pairs]


def run_training(
    config: Dict,
    backend_name: str,
    features_dir: Optional[str] = None,
    results_dir: str = "results",
    device: Optional[str] = None,
) -> Dict:
    """Run the full SVM + MLP training + evaluation."""
    fdir = _features_dir(backend_name, features_dir)
    blob = load_features(fdir)

    X_train, y_train = blob["X_train"], blob["y_train"]
    X_val, y_val = blob["X_val"], blob["y_val"]
    X_test, y_test = blob["X_test"], blob["y_test"]
    scaler = blob["scaler"]
    class_map = blob["class_map"]

    if X_train.shape[0] == 0:
        raise RuntimeError(
            f"No training data in {fdir}. Run extract_features first."
        )

    target_names = _target_names_from_class_map(class_map)
    num_classes = max(int(np.max(y_train)) + 1, len(class_map))

    Xtr = _scale(scaler, X_train)
    Xva = _scale(scaler, X_val)
    Xte = _scale(scaler, X_test)
    _logger.info("Feature shapes: train=%s val=%s test=%s", Xtr.shape, Xva.shape, Xte.shape)

    os.makedirs(results_dir, exist_ok=True)

    train_cfg = config.get("training", {}) or {}
    epochs = int(train_cfg.get("epochs", 100))
    batch_size = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-2))
    patience = int(train_cfg.get("early_stop_patience", 10))

    summary: Dict = {"backend": backend_name}

    # ------------------------------------------------------------------
    # SVM
    # ------------------------------------------------------------------
    print(_format_section("Training RBF SVM"))
    svm = train_svm(Xtr, y_train)
    svm_path = os.path.join(results_dir, f"svm_{backend_name}.joblib")
    joblib.dump(svm, svm_path)
    _logger.info("Saved SVM to %s", svm_path)

    svm_train_scores = svm_scores(svm, Xtr)
    svm_val_scores = svm_scores(svm, Xva)
    svm_test_scores = svm_scores(svm, Xte)

    summary["svm"] = {
        "train": evaluate_split("train", "SVM", y_train, svm_train_scores),
        "val": evaluate_split("val", "SVM", y_val, svm_val_scores),
        "test": evaluate_split("test", "SVM", y_test, svm_test_scores),
    }
    detailed_report(
        "test",
        "SVM",
        y_test,
        svm_test_scores.argmax(axis=1) if svm_test_scores.size else np.array([], dtype=int),
        target_names=target_names,
    )
    summary["svm"]["inference_ms_per_clip_test"] = _measure_inference_time(
        lambda x: svm_scores(svm, x), Xte
    )

    # ------------------------------------------------------------------
    # MLP
    # ------------------------------------------------------------------
    print(_format_section("Training MLP"))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp, _meta = train_mlp(
        Xtr,
        y_train,
        Xva,
        y_val,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        early_stop_patience=patience,
        device=device,
    )
    mlp_path = os.path.join(results_dir, f"mlp_{backend_name}.pt")
    torch.save(
        {
            "state_dict": mlp.state_dict(),
            "input_dim": Xtr.shape[1],
            "num_classes": num_classes,
            "class_names": target_names,
            "hidden": (256, 128),
            "dropout": 0.5,
            "backend": backend_name,
        },
        mlp_path,
    )
    _logger.info("Saved MLP to %s", mlp_path)

    mlp_train_scores = mlp_scores(mlp, Xtr, device=device)
    mlp_val_scores = mlp_scores(mlp, Xva, device=device)
    mlp_test_scores = mlp_scores(mlp, Xte, device=device)

    summary["mlp"] = {
        "train": evaluate_split("train", "MLP", y_train, mlp_train_scores),
        "val": evaluate_split("val", "MLP", y_val, mlp_val_scores),
        "test": evaluate_split("test", "MLP", y_test, mlp_test_scores),
    }
    detailed_report(
        "test",
        "MLP",
        y_test,
        mlp_test_scores.argmax(axis=1) if mlp_test_scores.size else np.array([], dtype=int),
        target_names=target_names,
    )
    mlp.to("cpu")
    summary["mlp"]["inference_ms_per_clip_test"] = _measure_inference_time(
        lambda x: mlp_scores(mlp, x, device="cpu"), Xte
    )

    # ------------------------------------------------------------------
    # save shared sidecars
    # ------------------------------------------------------------------
    joblib.dump(scaler, os.path.join(results_dir, f"scaler_{backend_name}.joblib"))
    with open(os.path.join(results_dir, "class_map.json"), "w") as f:
        json.dump(class_map, f, indent=2)

    # Final summary.
    print(_format_section("Summary"))
    for clf in ("svm", "mlp"):
        m = summary[clf]
        print(
            f"  {clf.upper():3s}  "
            f"train top1={m['train']['top1']:.3f}  "
            f"val top1={m['val']['top1']:.3f}  "
            f"test top1={m['test']['top1']:.3f}  "
            f"test top3={m['test']['top3']:.3f}  "
            f"test f1m={m['test']['f1_macro']:.3f}  "
            f"inf={m['inference_ms_per_clip_test']:.2f} ms/clip"
        )
    return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train RBF SVM + MLP on pre-extracted flow features."
    )
    parser.add_argument(
        "--backend",
        choices=("farneback", "raft"),
        required=True,
        help="Optical-flow backend used to produce the features.",
    )
    parser.add_argument(
        "--config",
        default="configs/wlasl10.yaml",
        help="YAML config (defaults to configs/wlasl10.yaml).",
    )
    parser.add_argument(
        "--features-dir",
        default=None,
        help="Override features dir (default: data/features/{backend}).",
    )
    parser.add_argument(
        "--results-dir", default="results", help="Where to write checkpoints."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for MLP (cpu / cuda / mps). Default: auto.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run_training(
        config=cfg,
        backend_name=args.backend,
        features_dir=args.features_dir,
        results_dir=args.results_dir,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
