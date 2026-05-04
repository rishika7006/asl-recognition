# Evaluation results

Everything in this folder is what the proposal §4 (Evaluation) asks for —
accuracy / precision / recall / F1, per-class metrics, per-clip latency,
flow-quality visualization, robustness across lighting, and live-demo
qualitative results. Re-generate any of these by running the scripts in
`src/eval/` from the project root.

## Files

| file | purpose | regenerate with |
|---|---|---|
| `comparison_table.md` | Headline backend × classifier table (top-1, top-3, F1, latency). | `python -m src.eval.compare_backends` |
| `comparison_table.csv` | Same data in CSV for spreadsheet import. | (same) |
| `per_class_metrics.md` | Sklearn `classification_report` per (backend, classifier) on the test split. | (same) |
| `confusion_matrices.png` | 2×2 grid of test-split confusion matrices for FB-SVM, FB-MLP, RAFT-SVM, RAFT-MLP. | (same) |
| `lighting_robustness.md` | Good-light vs low-light accuracy on the user_videos clips, plus classifier-independent feature cosine similarity per class. | `python -m src.eval.eval_lighting` |
| `flow_visualization.png` | Hand crop + Farnebäck flow + RAFT flow over 4 frame pairs of `computer_good_light.mov`. Hue = direction, brightness = magnitude. | `python -m src.eval.viz_flow` |
| `live_demo_score_template.md` | Blank score sheet for the report's live-demo qualitative evaluation. | (manual) |
| `computer_demo.mov` | Screen recording of the live demo predicting the *computer* sign. Reference media for the report. | (manual) |

## Headline numbers (as of last run)

```
| Backend   | Classifier | train top-1 | val top-1 | test top-1 | test top-3 | test F1 | ms/clip |
|-----------|------------|-------------|-----------|------------|------------|---------|---------|
| Farnebäck | SVM        | 0.857       | 0.095     | 0.231      | 0.385      | 0.134   | ~1.13   |
| Farnebäck | MLP        | 0.990       | 0.190     | 0.538      | 0.846      | 0.517   | ~0.05   |
| RAFT      | SVM        | 0.800       | 0.190     | 0.308      | 0.308      | 0.186   | ~1.02   |
| RAFT      | MLP        | 0.848       | 0.190     | 0.308      | 0.462      | 0.307   | ~0.06   |
```

**Best model:** Farnebäck-MLP — highest test top-1 (0.538), highest test
top-3 (0.846), highest F1 (0.517), and the lowest classifier-side latency.

## How the test set was scored

Test split: 13 WLASL clips drawn from the original `nslt_100.json`
test bucket of the 10 chosen classes. Note: some classes have no test
clips at all (book, chair) and others have only 1 — single-sample
accuracies are noisy. The macro F1 and confusion matrices are more
informative than the headline accuracy on this small split.

User-recorded clips (20 total, 2 per class — `_good_light` /
`_low_light`) were added to the **training** split during retraining,
not test. So all test numbers above measure WLASL→WLASL generalization,
not user-level recall.

## Reproducing live FPS

The proposal asks for FPS / latency. Three latency numbers to cite:

1. **Classifier inference**, from `comparison_table.md` (`ms/clip`).
   This is the SVM/MLP forward pass only.
2. **Per-pair flow latency**, measured during `tests/smoke_raft.py`:
   Farnebäck ~5 ms/pair (CPU), RAFT ~16 ms/pair (CPU at 96×96, 6 iter).
3. **End-to-end live FPS**, shown in the `app.py` browser overlay:
   ~30 fps camera, ~12 preds/sec Farnebäck, ~6 preds/sec RAFT on M4 MPS.

## Notes for the writeup

- The dataset is genuinely small (85 WLASL train + 20 user clips for 10
  classes, ~10 per class). The train/val gap (~99% / ~19%) reflects this.
- RAFT did NOT beat Farnebäck on this task. That's a real finding worth
  citing — once flow is grid-pooled into 144 cells, the classical-vs-deep
  flow-estimator quality gap mostly washes out. The argument for RAFT
  here is reproducibility and motion-faithfulness, not accuracy.
- Adding 20 user clips bumped Farnebäck-MLP test top-1 by **+23 absolute
  points** (0.308 → 0.538), confirming that the dominant bottleneck is
  signer-distribution mismatch, not architecture.
