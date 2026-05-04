"""Real-time ASL recognition demo with parallel Farnebäck + RAFT backends.

Architecture:

    +--------------------+      +-------------------------+
    |  Flask streamer    |      |  Backend worker (xN)    |
    |  thread            |      |                         |
    |                    |      |  - reads cropped buffer |
    |  - reads webcam    |      |  - computes new flow    |
    |  - MediaPipe bbox  | ---> |  - updates 15-pair stack|
    |  - smooths bbox    |      |  - scales + classifies  |
    |  - draws overlays  |      |  - publishes prediction |
    |  - encodes JPEG    |      +-------------------------+
    +--------------------+         (one per backend)
                                ^
                                |
                +----------------+----------------+
                |  Shared rolling buffer (lock)   |
                |  16 cropped 96x96 RGB frames    |
                +----------------------------------+

Both backends run on the SAME live webcam. Each maintains its own
incremental 15-pair feature stack — when a new cropped frame arrives,
each worker only computes flow for the *newest* pair and shifts the
stack, so RAFT doesn't have to re-run on the whole window every tick.

If a checkpoint for either backend is missing (e.g. RAFT training
hasn't finished yet), that backend's pane shows "model not loaded" and
the rest of the demo keeps running.

Endpoints:
    /              - UI page
    /video_feed    - MJPEG stream of the annotated webcam
    /predictions   - JSON {farneback: {...}, raft: {...}, fps: {...}}
"""

from __future__ import annotations

import os

# OpenCV's macOS AVFoundation backend tries to pop the camera-permission
# dialog from inside cv2.VideoCapture(0). That requires the macOS main
# run loop, which Flask worker threads don't have, so the dialog fails.
# We set this env var BEFORE importing cv2 so OpenCV skips its auth
# request entirely; we instead verify access ourselves on the main
# thread at startup (see __main__).
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import cv2  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from flask import Flask, Response, jsonify, render_template_string  # noqa: E402

from src.extract_features import grid_pool_flow
from src.flow_backends import get_backend
from src.hand_crop import HandCropper
from src.train import FlowMLP, svm_scores

# ---------------------------------------------------------------------
# config + paths
# ---------------------------------------------------------------------

CONFIG_PATH = "configs/wlasl10.yaml"
RESULTS_DIR = "results"
WLASL_CLASS_LIST = "wlasl_class_list.txt"
CLASS_MAP_PATH = os.path.join(RESULTS_DIR, "class_map.json")

# Which webcam to open. Override with ``ASL_CAMERA_INDEX=1 python app.py``
# (or 2, 3, …) when macOS Continuity Camera is grabbing your iPhone at 0.
CAMERA_INDEX = int(os.environ.get("ASL_CAMERA_INDEX", "0"))

# Stride between the camera frames we push into the cropped buffer.
# Camera typically runs ~30 fps; stride=2 keeps the buffer at ~15 fps,
# matching the temporal density of features used at training time.
CAMERA_STRIDE = 2

# Confidence threshold below which a backend's prediction is suppressed.
MIN_CONFIDENCE = 0.45

# How often each worker tries to update its prediction (seconds).
# Workers also gate on "is there a new frame in the buffer?".
FB_TICK_SECONDS = 1.0 / 12.0   # ~12 attempted preds/s
RAFT_TICK_SECONDS = 1.0 / 6.0  # ~6 attempted preds/s

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
_logger = logging.getLogger("app")


# ---------------------------------------------------------------------
# class name resolution
# ---------------------------------------------------------------------

def _load_wlasl_text_map(path: str) -> Dict[int, str]:
    """Return {wlasl_class_id: gloss_text}. Tab-separated file."""
    out: Dict[int, str] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                try:
                    out[int(parts[0])] = parts[1]
                except ValueError:
                    pass
    return out


def _resolve_class_names(
    class_map: Dict[str, int], wlasl_text: Dict[int, str]
) -> List[str]:
    """Build a list[str] of length K, indexed by REMAPPED class index.

    ``class_map`` maps WLASL class id (str key) -> remapped index
    (0..K-1). We invert that and look up the gloss text.
    """
    inv = sorted(class_map.items(), key=lambda kv: int(kv[1]))
    names: List[str] = []
    for orig_id_str, _ in inv:
        try:
            wlasl_id = int(orig_id_str)
            names.append(wlasl_text.get(wlasl_id, f"id{wlasl_id}"))
        except ValueError:
            names.append(orig_id_str)
    return names


# ---------------------------------------------------------------------
# shared state — protected by `state.lock`
# ---------------------------------------------------------------------

class SharedState:
    """Holds the rolling cropped buffer and the latest predictions.

    Camera generator is the producer for ``cropped_buffer`` and
    ``display_meta``; each backend worker is a consumer of the buffer
    and a producer of its own slot in ``predictions``. All access is
    serialized through a single re-entrant lock to keep the demo
    simple — contention is negligible at these rates.
    """

    def __init__(self, num_frames: int) -> None:
        self.num_frames = num_frames
        self.lock = threading.RLock()

        # Shared rolling buffer of CROPPED 96x96 RGB frames.
        # Each entry is (counter, frame). The counter monotonically
        # increases so workers can detect "new frames since last tick"
        # without comparing arrays.
        self.cropped_buffer: deque[Tuple[int, np.ndarray]] = deque(
            maxlen=num_frames
        )
        self.frame_counter: int = 0

        # bbox in display-frame coordinates (x1, y1, x2, y2). For UI
        # drawing only — workers don't read this.
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None

        # Latest publishable prediction per backend.
        self.predictions: Dict[str, Dict[str, Any]] = {
            "farneback": _default_pred(),
            "raft": _default_pred(),
        }

        # FPS measurements.
        self.fps: Dict[str, float] = {
            "camera": 0.0,
            "farneback": 0.0,
            "raft": 0.0,
        }

        self.shutdown = threading.Event()


def _default_pred() -> Dict[str, Any]:
    return {
        "label": "—",
        "confidence": 0.0,
        "status": "warming up",  # "warming up" | "ok" | "low conf" | "no model"
        "classifier": "—",
    }


# ---------------------------------------------------------------------
# classifier loading / inference
# ---------------------------------------------------------------------

class BackendClassifier:
    """Wraps the trained classifier (MLP preferred, SVM fallback) for one backend.

    Loads the trained MLP from ``results/mlp_{backend}.pt`` if available,
    otherwise the SVM from ``results/svm_{backend}.joblib``. Either way,
    exposes a single ``predict(features_flat)`` method that returns
    ``(label_index, confidence, classifier_name)``.
    """

    def __init__(self, backend_name: str, class_names: List[str]) -> None:
        self.backend_name = backend_name
        self.class_names = class_names
        self.scaler = None
        self.mlp: Optional[FlowMLP] = None
        self.svm = None
        self.kind = "none"

        scaler_path = os.path.join(RESULTS_DIR, f"scaler_{backend_name}.joblib")
        mlp_path = os.path.join(RESULTS_DIR, f"mlp_{backend_name}.pt")
        svm_path = os.path.join(RESULTS_DIR, f"svm_{backend_name}.joblib")

        if not os.path.exists(scaler_path):
            _logger.warning(
                "%s: no scaler at %s — backend will show 'no model'",
                backend_name,
                scaler_path,
            )
            return
        self.scaler = joblib.load(scaler_path)

        if os.path.exists(mlp_path):
            blob = torch.load(mlp_path, map_location="cpu", weights_only=False)
            mlp = FlowMLP(
                input_dim=blob["input_dim"],
                num_classes=blob["num_classes"],
                hidden=tuple(blob["hidden"]),
                dropout=blob["dropout"],
            )
            mlp.load_state_dict(blob["state_dict"])
            mlp.eval()
            self.mlp = mlp
            self.kind = "MLP"
            _logger.info("%s: loaded MLP from %s", backend_name, mlp_path)
        elif os.path.exists(svm_path):
            self.svm = joblib.load(svm_path)
            self.kind = "SVM"
            _logger.info("%s: loaded SVM from %s", backend_name, svm_path)
        else:
            _logger.warning(
                "%s: no classifier checkpoint found in %s",
                backend_name,
                RESULTS_DIR,
            )

    @property
    def loaded(self) -> bool:
        return self.kind != "none"

    def predict(self, features_flat: np.ndarray) -> Tuple[int, float, str]:
        """Return ``(class_idx, confidence, classifier_name)``."""
        x = self.scaler.transform(features_flat.reshape(1, -1)).astype(np.float32)
        if self.mlp is not None:
            with torch.no_grad():
                t = torch.from_numpy(x)
                logits = self.mlp(t)
                probs = F.softmax(logits, dim=1).numpy()[0]
            idx = int(np.argmax(probs))
            return idx, float(probs[idx]), "MLP"
        if self.svm is not None:
            # Use decision_function + softmax (matches src.train.svm_scores).
            # The trained SVM has probability=False intentionally.
            probs = svm_scores(self.svm, x)[0]
            idx = int(np.argmax(probs))
            return idx, float(probs[idx]), "SVM"
        raise RuntimeError("predict called on backend with no loaded model")


# ---------------------------------------------------------------------
# Backend workers
# ---------------------------------------------------------------------

class BackendWorker(threading.Thread):
    """Background thread that runs one optical-flow backend.

    Each worker maintains its own incremental 15-pair feature stack so
    it doesn't recompute flow on the whole 16-frame window every tick.
    When a new cropped frame is added to the shared buffer, the worker
    computes flow only for the (prev, new) pair, pools it to a 12x12
    grid, appends to its feature deque, and (when the deque is full)
    runs the classifier.
    """

    def __init__(
        self,
        backend_name: str,
        device: str,
        state: SharedState,
        classifier: BackendClassifier,
        config: Dict,
        tick_seconds: float,
    ) -> None:
        super().__init__(name=f"worker:{backend_name}", daemon=True)
        self.backend_name = backend_name
        self.state = state
        self.classifier = classifier
        self.tick_seconds = tick_seconds

        # Build flow estimator on requested device.
        self.backend_module = get_backend(backend_name)
        self.estimator = self.backend_module.make_estimator(device=device)

        ds = config["dataset"]
        self.num_pairs = int(ds["num_frames"]) - 1
        self.grid_size = int(ds.get("grid_size", 12))

        # Worker-local rolling stack of pooled flow features.
        # When length == num_pairs we have a full clip and can predict.
        self._features: deque[np.ndarray] = deque(maxlen=self.num_pairs)
        self._last_seen_counter: int = -1
        self._prev_frame: Optional[np.ndarray] = None

        # FPS tracking.
        self._pred_times: deque[float] = deque(maxlen=20)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        if not self.classifier.loaded:
            with self.state.lock:
                self.state.predictions[self.backend_name] = {
                    "label": "—",
                    "confidence": 0.0,
                    "status": "no model",
                    "classifier": "—",
                }
            _logger.info(
                "%s worker exiting: no classifier", self.backend_name
            )
            return

        while not self.state.shutdown.is_set():
            tick_start = time.perf_counter()
            self._step()
            elapsed = time.perf_counter() - tick_start
            sleep_for = max(0.0, self.tick_seconds - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _step(self) -> None:
        # Snapshot any new frames from the shared buffer under lock.
        new_frames = self._collect_new_frames()
        if not new_frames:
            return

        # For each new frame, compute the new flow pair against the prior
        # frame, pool, and append to the worker-local stack.
        for frame in new_frames:
            if self._prev_frame is None:
                # First frame ever — can't compute a pair yet.
                self._prev_frame = frame
                continue
            try:
                flow = self.backend_module.flow(
                    self.estimator, self._prev_frame, frame
                )
            except Exception as e:  # noqa: BLE001
                _logger.exception(
                    "%s flow() failed: %s", self.backend_name, e
                )
                self._prev_frame = frame
                continue
            try:
                pooled = grid_pool_flow(flow, self.grid_size)
            except Exception as e:  # noqa: BLE001
                _logger.exception(
                    "%s grid_pool_flow failed: %s", self.backend_name, e
                )
                self._prev_frame = frame
                continue
            self._features.append(pooled)
            self._prev_frame = frame

        if len(self._features) < self.num_pairs:
            with self.state.lock:
                self.state.predictions[self.backend_name] = {
                    "label": "—",
                    "confidence": 0.0,
                    "status": (
                        f"warming up ({len(self._features)}/{self.num_pairs})"
                    ),
                    "classifier": self.classifier.kind,
                }
            return

        # Full window -> classify.
        feats_flat = np.concatenate(list(self._features), axis=0).astype(
            np.float32, copy=False
        )
        try:
            idx, conf, clf_name = self.classifier.predict(feats_flat)
        except Exception as e:  # noqa: BLE001
            _logger.exception(
                "%s classifier.predict failed: %s", self.backend_name, e
            )
            return

        label = (
            self.classifier.class_names[idx]
            if 0 <= idx < len(self.classifier.class_names)
            else f"class_{idx}"
        )

        if conf < MIN_CONFIDENCE:
            status = "low conf"
            display_label = "—"
        else:
            status = "ok"
            display_label = label.upper()

        # FPS bookkeeping.
        now = time.perf_counter()
        self._pred_times.append(now)
        fps = _fps_from_times(self._pred_times)

        with self.state.lock:
            self.state.predictions[self.backend_name] = {
                "label": display_label,
                "confidence": float(conf),
                "status": status,
                "classifier": clf_name,
                "raw_label": label,
            }
            self.state.fps[self.backend_name] = fps

    def _collect_new_frames(self) -> List[np.ndarray]:
        """Drain frames from the shared buffer that we haven't seen yet."""
        new_frames: List[np.ndarray] = []
        with self.state.lock:
            for counter, frame in self.state.cropped_buffer:
                if counter > self._last_seen_counter:
                    new_frames.append(frame)
                    self._last_seen_counter = counter
        return new_frames


# ---------------------------------------------------------------------
# bbox smoothing
# ---------------------------------------------------------------------

def _ema_bbox(
    prev: Optional[Tuple[int, int, int, int]],
    new: Optional[Tuple[int, int, int, int]],
    alpha: float = 0.6,
) -> Optional[Tuple[int, int, int, int]]:
    """EMA smoothing for the bbox coordinates to reduce jitter."""
    if new is None:
        # Decay back to None more slowly so brief detection misses don't
        # remove the box. Easiest behavior: keep the previous box.
        return prev
    if prev is None:
        return new
    return tuple(
        int(round(alpha * n + (1.0 - alpha) * p))
        for n, p in zip(new, prev)
    )  # type: ignore[return-value]


# ---------------------------------------------------------------------
# camera streamer
# ---------------------------------------------------------------------

def _fps_from_times(times: deque) -> float:
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    if span <= 0:
        return 0.0
    return (len(times) - 1) / span


def make_streamer(
    state: SharedState,
    config: Dict,
):
    """Build the MJPEG generator that reads the webcam and pushes
    cropped frames into the shared buffer."""
    crop_size = int(config["dataset"].get("crop_size", 96))
    cropper = HandCropper()

    def stream():
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            _logger.error(
                "Could not open webcam at index %d (cv2.VideoCapture)",
                CAMERA_INDEX,
            )
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Log the opened resolution so the user can tell which physical
        # camera macOS picked (Continuity Camera at e.g. 1920x1080 vs
        # built-in FaceTime HD at 1280x720).
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _logger.info(
            "Opened camera index=%d at %dx%d", CAMERA_INDEX, w, h
        )

        cam_times: deque[float] = deque(maxlen=30)
        bbox_smoothed: Optional[Tuple[int, int, int, int]] = None
        cam_idx = 0
        try:
            while not state.shutdown.is_set():
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break
                cam_idx += 1
                cam_times.append(time.perf_counter())

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Per-frame bbox + smoothing (cheap-ish).
                bbox_raw = cropper.bbox_for_frame(frame_rgb)
                bbox_smoothed = _ema_bbox(bbox_smoothed, bbox_raw, alpha=0.6)

                # Push to cropped buffer at stride.
                if cam_idx % CAMERA_STRIDE == 0 and bbox_smoothed is not None:
                    cropped = cropper.crop_clip(
                        [frame_rgb], bbox_smoothed, target_size=crop_size
                    )[0]
                    with state.lock:
                        state.frame_counter += 1
                        state.cropped_buffer.append(
                            (state.frame_counter, cropped.copy())
                        )

                # Snapshot predictions for overlay.
                with state.lock:
                    state.last_bbox = bbox_smoothed
                    preds = json.loads(json.dumps(state.predictions))
                    fps_cam = _fps_from_times(cam_times)
                    state.fps["camera"] = fps_cam
                    fps_fb = state.fps["farneback"]
                    fps_raft = state.fps["raft"]

                # Draw overlays on a BGR copy for streaming.
                annotated = frame_bgr.copy()
                _draw_overlay(
                    annotated,
                    bbox_smoothed,
                    preds,
                    fps_cam=fps_cam,
                    fps_fb=fps_fb,
                    fps_raft=fps_raft,
                )

                ok_jpg, jpg = cv2.imencode(
                    ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                if not ok_jpg:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpg.tobytes()
                    + b"\r\n"
                )
        finally:
            cap.release()
            cropper.close()

    return stream


def _draw_overlay(
    frame_bgr: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    preds: Dict[str, Dict[str, Any]],
    fps_cam: float,
    fps_fb: float,
    fps_raft: float,
) -> None:
    """Mutates frame_bgr in place: draws bbox + per-backend prediction text."""
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    h, w = frame_bgr.shape[:2]

    # Top: backend predictions.
    fb = preds.get("farneback", _default_pred())
    raft = preds.get("raft", _default_pred())
    agree = (
        fb.get("label") not in ("—", "")
        and fb.get("label") == raft.get("label")
    )

    # NOTE: cv2.putText only supports ASCII. Use '---' (not '—') as the
    # no-prediction placeholder so the OpenCV font doesn't replace the
    # em-dash with '?' / '???'. The HTML panes still render '—' just fine.
    fb_label = fb.get("label", "---")
    if fb_label == "—":
        fb_label = "---"
    raft_label = raft.get("label", "---")
    if raft_label == "—":
        raft_label = "---"
    fb_text = (
        f"FB:   {fb_label:<14s}  "
        f"{int(fb.get('confidence', 0.0) * 100):3d}%  "
        f"({fb.get('status', '---')})"
    )
    raft_text = (
        f"RAFT: {raft_label:<14s}  "
        f"{int(raft.get('confidence', 0.0) * 100):3d}%  "
        f"({raft.get('status', '---')})"
    )
    fps_text = (
        f"cam {fps_cam:4.1f} fps | fb {fps_fb:3.1f} preds/s | "
        f"raft {fps_raft:3.1f} preds/s"
    )

    # Translucent backdrop so text is readable over varied backgrounds.
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0, frame_bgr)

    color_fb = (0, 255, 200)
    color_raft = (180, 200, 255)
    color_agree = (0, 255, 0)
    cv2.putText(
        frame_bgr,
        fb_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color_agree if agree else color_fb,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        raft_text,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color_agree if agree else color_raft,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        fps_text,
        (10, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------
# Flask wiring
# ---------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>ASL Recognition — Farnebäck vs RAFT</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                         Roboto, Helvetica, Arial, sans-serif;
            background: #0e0e10; color: #f5f5f5; text-align: center;
            padding: 24px; margin: 0;
        }
        h1 { color: #bb86fc; margin-bottom: 4px; }
        .sub { color: #aaa; margin-bottom: 24px; }
        .video {
            margin: 0 auto 20px auto;
            border: 4px solid #2a2a2e; border-radius: 12px;
            display: inline-block; overflow: hidden;
            box-shadow: 0 6px 24px rgba(0,0,0,0.4);
        }
        .panes {
            display: flex; justify-content: center;
            gap: 16px; flex-wrap: wrap; margin-top: 12px;
        }
        .pane {
            background: #1a1a1d; border-radius: 12px; padding: 18px 22px;
            min-width: 280px; text-align: left;
        }
        .pane h2 { margin: 0 0 8px 0; font-size: 1.1rem; color: #bb86fc; }
        .label { font-size: 1.8rem; font-weight: 700; color: #03dac6; }
        .conf { font-size: 0.95rem; color: #888; margin-top: 4px; }
        .meta { font-size: 0.8rem; color: #666; margin-top: 6px; }
        .agree-yes .label { color: #00ff80; }
        .agree-no .label { color: #ffcc66; }
        .fps {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.85rem; color: #888; margin-top: 16px;
        }
        .legend { font-size: 0.8rem; color: #666; margin-top: 6px; }
    </style>
</head>
<body>
    <h1>Real-Time ASL Recognition</h1>
    <div class="sub">Farnebäck (classical) vs RAFT (deep learning)</div>

    <div class="video">
        <img src="/video_feed" width="640" height="480">
    </div>

    <div class="panes">
        <div class="pane" id="pane-fb">
            <h2>Farnebäck</h2>
            <div class="label" id="fb-label">—</div>
            <div class="conf"  id="fb-conf">confidence —</div>
            <div class="meta"  id="fb-meta">classifier — • status —</div>
        </div>
        <div class="pane" id="pane-raft">
            <h2>RAFT</h2>
            <div class="label" id="raft-label">—</div>
            <div class="conf"  id="raft-conf">confidence —</div>
            <div class="meta"  id="raft-meta">classifier — • status —</div>
        </div>
    </div>

    <div class="fps" id="fps">camera — | fb — | raft —</div>
    <div class="legend">Green = algorithms agree || amber = algorithmsdisagree</div>

    <script>
    function fmtPct(p) { return Math.round((p || 0) * 100) + "%"; }
    function update() {
        fetch("/predictions").then(r => r.json()).then(d => {
            const fb = d.predictions.farneback || {};
            const raft = d.predictions.raft || {};
            const agree =
                fb.label && raft.label && fb.label !== "—" &&
                fb.label === raft.label;

            document.getElementById("fb-label").innerText  = fb.label || "—";
            document.getElementById("fb-conf").innerText   =
                "confidence " + fmtPct(fb.confidence);
            document.getElementById("fb-meta").innerText   =
                "classifier " + (fb.classifier || "—") +
                " • status " + (fb.status || "—");

            document.getElementById("raft-label").innerText  = raft.label || "—";
            document.getElementById("raft-conf").innerText   =
                "confidence " + fmtPct(raft.confidence);
            document.getElementById("raft-meta").innerText   =
                "classifier " + (raft.classifier || "—") +
                " • status " + (raft.status || "—");

            for (const id of ["pane-fb", "pane-raft"]) {
                const el = document.getElementById(id);
                el.classList.remove("agree-yes", "agree-no");
                if (fb.label && raft.label && fb.label !== "—" && raft.label !== "—") {
                    el.classList.add(agree ? "agree-yes" : "agree-no");
                }
            }

            const fps = d.fps || {};
            document.getElementById("fps").innerText =
                "camera " + (fps.camera || 0).toFixed(1) + " fps " +
                "| fb " + (fps.farneback || 0).toFixed(1) + " preds/s " +
                "| raft " + (fps.raft || 0).toFixed(1) + " preds/s";
        }).catch(() => {});
    }
    setInterval(update, 250);
    update();
    </script>
</body>
</html>
"""

# ---------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------

def _load_config() -> Dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _pick_device_for(name: str) -> str:
    """Choose where each backend runs.

    Farnebäck is OpenCV CPU only. RAFT prefers MPS on Apple Silicon when
    available, with a CPU fallback if PyTorch hasn't built with MPS or
    the user explicitly disabled it.
    """
    if name == "farneback":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_app() -> Flask:
    app = Flask(__name__)
    config = _load_config()

    # Class name resolution.
    wlasl_text = _load_wlasl_text_map(WLASL_CLASS_LIST)
    if os.path.exists(CLASS_MAP_PATH):
        with open(CLASS_MAP_PATH, "r") as f:
            class_map = json.load(f)
        class_names = _resolve_class_names(class_map, wlasl_text)
    else:
        _logger.warning(
            "No class_map.json at %s — workers will run with empty label table",
            CLASS_MAP_PATH,
        )
        class_names = []

    state = SharedState(num_frames=int(config["dataset"]["num_frames"]))

    # Spin up one worker per backend.
    classifiers = {
        n: BackendClassifier(n, class_names) for n in ("farneback", "raft")
    }
    workers = []
    for name, tick in (
        ("farneback", FB_TICK_SECONDS),
        ("raft", RAFT_TICK_SECONDS),
    ):
        device = _pick_device_for(name)
        try:
            w = BackendWorker(
                backend_name=name,
                device=device,
                state=state,
                classifier=classifiers[name],
                config=config,
                tick_seconds=tick,
            )
        except Exception as e:  # noqa: BLE001
            _logger.exception(
                "Could not construct %s worker on device=%s: %s",
                name,
                device,
                e,
            )
            with state.lock:
                state.predictions[name] = {
                    "label": "—",
                    "confidence": 0.0,
                    "status": f"backend init failed: {e}",
                    "classifier": "—",
                }
            continue
        w.start()
        workers.append(w)
        _logger.info("Started %s worker on device=%s", name, device)

    streamer = make_streamer(state, config)

    @app.route("/")
    def index():  # type: ignore[unused-ignore]
        return render_template_string(HTML_TEMPLATE)

    @app.route("/video_feed")
    def video_feed():  # type: ignore[unused-ignore]
        return Response(
            streamer(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    @app.route("/predictions")
    def predictions():  # type: ignore[unused-ignore]
        with state.lock:
            return jsonify(
                {
                    "predictions": dict(state.predictions),
                    "fps": dict(state.fps),
                }
            )

    @app.route("/healthz")
    def healthz():  # type: ignore[unused-ignore]
        with state.lock:
            return jsonify(
                {
                    "buffer_len": len(state.cropped_buffer),
                    "frame_counter": state.frame_counter,
                    "predictions": dict(state.predictions),
                    "fps": dict(state.fps),
                }
            )

    return app


def _probe_cameras(max_index: int = 4) -> List[Tuple[int, int, int]]:
    """Probe camera indices 0..max_index-1; return (index, width, height)
    for those that open and successfully return a frame.
    """
    found: List[Tuple[int, int, int]] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            found.append((i, w, h))
        cap.release()
    return found


def _verify_camera_access_or_exit() -> None:
    """Open the configured camera on the main thread, verify it actually
    delivers a frame, and print which physical device showed up.

    On macOS, ``cv2.VideoCapture(N)`` from a worker thread fails silently
    if the host app hasn't been granted camera access — and even when
    permission is granted, Continuity Camera will hijack index 0 with
    your iPhone if it's nearby. This probe makes both situations visible.
    """
    found = _probe_cameras(max_index=4)
    if not found:
        print(
            "\n[ERROR] No working webcam found at indices 0..3.\n"
            "  On macOS: open System Settings → Privacy & Security → Camera,\n"
            "  enable the terminal app you are running this from (Terminal,\n"
            "  iTerm, VS Code, Cursor, etc.), then fully quit and relaunch\n"
            "  that terminal before re-running `python app.py`.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Available cameras (index → resolution):")
    for idx, w, h in found:
        marker = "  <-- selected" if idx == CAMERA_INDEX else ""
        print(f"  [{idx}] {w}x{h}{marker}")

    if CAMERA_INDEX not in {idx for idx, _, _ in found}:
        print(
            f"\n[ERROR] Configured CAMERA_INDEX={CAMERA_INDEX} did not "
            f"open. Pick one from the list above and re-run with e.g.\n"
            f"  ASL_CAMERA_INDEX={found[0][0]} python app.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        "\nIf the wrong device is selected (e.g. iPhone Continuity "
        "Camera at 1920x1080 instead of the built-in FaceTime HD), "
        "re-run with:\n  ASL_CAMERA_INDEX=<idx> python app.py\n"
        "or disable Continuity Camera on your iPhone:\n"
        "  Settings → General → AirPlay & Continuity → Continuity Camera → off\n"
    )


if __name__ == "__main__":
    _verify_camera_access_or_exit()
    application = build_app()
    print("Starting web server. Open http://localhost:5001 in your browser.")
    application.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
