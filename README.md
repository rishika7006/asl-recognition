# ASL Real-Time Sign Recognition — Group 14

Real-time American Sign Language recognition from a webcam, using optical flow 
(Farnebäck and RAFT) and a lightweight classifier. Runs side-by-side parallel inference 
on both flow estimators so the live demo can compare classical vs deep flow estimation
on the same camera stream.

## How to run

### 1. Prerequisites

- macOS or Linux (Mac M-series tested)
- **Python 3.11** (3.9 fails on the MediaPipe wheel — install via `brew install python@3.11` if needed)
- A Kaggle account (for the dataset)
- ~7 GB free disk

### 2. Install

```bash
git clone https://github.com/TanviDeore/ASL_Recognition_Grp14.git
cd ASL_Recognition_Grp14

python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> **Note:** Steps 3–5 are only needed to retrain. The trained checkpoints in `results/` are already included, so to just run the live demo skip ahead to Step 6.

### 3. Get the dataset

Download the WLASL-processed dataset zip from
<https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed>, then:

```bash
mkdir -p data/raw
unzip -q ~/Downloads/archive.zip -d data/raw/
# verify: should print ~12000
find data/raw -name "*.mp4" | wc -l
```

### 3a. Get the user-recorded clips

The team-recorded clips (10 classes × 2 lighting variants) can be downloaded as `user_videos.zip` from:

<https://utdallas.box.com/s/2sclprxdtlv7yc26xmfxb2nx14kybh04>

and then unzip into `data/user_videos/`:

```bash
mkdir -p data/user_videos
unzip -q ~/Downloads/user_videos.zip -d data/user_videos/
# verify: should print 20 (10 classes × 2 clips)
find data/user_videos -name "*.mov" | wc -l
```

### 4. Extract features (both estimators)

```bash
python -m src.extract_features --backend farneback           # ~1 min
python -m src.extract_features --backend raft --device mps   # ~1.5 min on M-series; use --device cpu otherwise
```

This writes feature `.npy` files + scaler under `data/features/{farneback,raft}/`.
User-recorded clips in `data/user_videos/<class>/` (downloaded in step 3a)
get auto-mixed into the train split.

### 5. Train

```bash
python -m src.train --backend farneback   # ~30 sec — saves results/{svm,mlp}_farneback.{joblib,pt}
python -m src.train --backend raft        # ~30 sec — saves results/{svm,mlp}_raft.{joblib,pt}
```

Each run prints train/val/test top-1, top-3, F1, confusion matrix, per-clip
inference latency.

### 6. Run the live demo

```bash
python app.py
# open http://localhost:5001
```

On first launch, macOS will prompt for camera permission for whichever
terminal launched python — grant it and **fully quit / relaunch the
terminal** before re-running.

If macOS Continuity Camera launches iPhone camera instead of the Mac webcam,
the startup probe lists all available cameras; pick the right index:

```bash
ASL_CAMERA_INDEX=1 python app.py
```

The demo runs Farnebäck and RAFT in parallel and shows side-by-side
predictions. Both panes turn green when the estimators agree.

### 7. (Optional) Regenerate evaluation artifacts

```bash
python -m src.eval.compare_backends   # comparison table + confusion matrix figure
python -m src.eval.eval_lighting      # good vs low light robustness analysis
python -m src.eval.viz_flow           # Farnebäck-vs-RAFT flow visualization figure
```

Outputs go into `eval_results/`.

## Repository layout

```
src/
  extract_features.py      hand-crop + window-aware flow extractor (both estimators)
  flow_backends.py         dispatcher
  farneback_backend.py     classical optical flow
  raft_backend.py          torchvision raft_small
  hand_crop.py             MediaPipe-based hand bbox + crop
  dataset.py               PyTorch Dataset with frame-level augmentation
  train.py                 RBF SVM + MLP training + evaluation
  eval/                    evaluation scripts
tests/                     synthetic-data smoke tests for both estimators
configs/wlasl10.yaml       pipeline config (top_k, crop size, grid size)
data/
  nslt_100.json            WLASL split metadata
  missing.txt              YouTube IDs no longer available
  user_videos/             team-recorded clips (10 classes × 2 lighting variants)
  features/                extracted feature .npy files
  raw/                     WLASL videos — TO BE DOWNLOADED
results/                   trained checkpoints + scalers
eval_results/              figures + tables + demo media
app.py                     Flask live-demo server
```

## Trained class set

The pipeline is configured for the 10 most-populated WLASL signs after
filtering missing videos:

```
0  book        5  chair
1  drink       6  who
2  computer    7  clothes
3  before      8  candy
4  go          9  cousin
```

Reference videos for each sign will be present under `data/raw/videos/` once the
dataset is downloaded.
