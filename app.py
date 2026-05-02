from flask import Flask, render_template_string, Response, jsonify
import os
import cv2
import torch
import numpy as np
from collections import deque
import yaml
import json
from src.extract_features import extract_flow_features
from src.train_simple import LightweightNN
import threading
import mediapipe as mp

app = Flask(__name__)

# Global variables to hold the latest prediction
latest_prediction = "Waiting for frames..."
prediction_lock = threading.Lock()
prediction_history = deque(maxlen=5)


def get_model_and_scaler():
    with open("configs/top10.yaml", "r") as f:
        config = yaml.safe_load(f)

    num_classes = config["dataset"]["top_k"]
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = LightweightNN(input_dim=1920, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("results/lightweight_nn.pt", map_location=device))
    model.eval()

    mean = np.load("data/features/scaler_mean.npy")
    std = np.load("data/features/scaler_std.npy")

    with open("data/features/class_map.json", "r") as f:
        class_map = json.load(f)
        
    inv_class_map_orig = {v: int(k) for k, v in class_map.items()}
    wlasl_text_map = {}
    if os.path.exists("wlasl_class_list.txt"):
        with open("wlasl_class_list.txt", "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    wlasl_text_map[int(parts[0])] = parts[1]
    
    inv_class_map = {}
    for model_id, orig_id in inv_class_map_orig.items():
        inv_class_map[model_id] = wlasl_text_map.get(orig_id, f"Sign ID {orig_id}")

    return model, mean, std, inv_class_map, device, config["dataset"]["num_frames"]


def generate_frames():
    global latest_prediction
    model, mean, std, inv_class_map, device, num_frames = get_model_and_scaler()
    buffer = deque(maxlen=num_frames)

    # Use cv2.CAP_AVFOUNDATION for Mac to explicitly request permissions sometimes works better
    cap = cv2.VideoCapture(0)
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw MediaPipe hand landmarks
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                )

        frame_resized = cv2.resize(frame_rgb, (128, 128))
        buffer.append(frame_resized)

        if len(buffer) == num_frames:
            frames_list = list(buffer)
            features = extract_flow_features(frames_list, grid_size=8)
            
            motion_energy = np.max(np.abs(features))
            
            if motion_energy < 0.3:
                with prediction_lock:
                    latest_prediction = "Waiting for sign..."
                prediction_history.clear()
            else:
                features_scaled = (features - mean) / std
    
                x_tensor = (
                    torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                )
    
                with torch.no_grad():
                    outputs = model(x_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    top_prob, top_idx = torch.max(probs, dim=1)
    
                class_id = top_idx.item()
                prob = top_prob.item()
    
                class_text = inv_class_map.get(class_id, f"Class {class_id}")
                print(f"Motion Energy: {motion_energy:.2f} | Prob: {prob*100:.1f}% | Top Class: {class_text}")
                
                if prob > 0.65:
                    prediction_history.append(class_text)
                else:
                    prediction_history.append("Waiting for sign...")
    
                # If the same class is predicted at least 3 times in the last 5 frames
                if len(prediction_history) == 5 and prediction_history.count(class_text) >= 3 and prob > 0.65:
                    with prediction_lock:
                        latest_prediction = f"{class_text.upper()} ({prob*100:.1f}%)"
                elif len(prediction_history) > 0 and prediction_history.count("Waiting for sign...") >= 3:
                    with prediction_lock:
                        latest_prediction = "Waiting for sign..."

            # Draw on frame as well
            cv2.putText(
                frame,
                latest_prediction,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        ret, buffer_img = cv2.imencode(".jpg", frame)
        frame_bytes = buffer_img.tobytes()

        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ASL Recognition - Realtime Demo</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #121212; color: #ffffff; text-align: center; padding: 2rem; }
        h1 { color: #bb86fc; }
        .video-container { margin: 2rem auto; border: 4px solid #333; border-radius: 8px; width: fit-content; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .prediction-box { background-color: #1e1e1e; padding: 1.5rem; border-radius: 8px; font-size: 2rem; font-weight: bold; color: #03dac6; margin-top: 1rem; }
    </style>
</head>
<body>
    <h1>Real-Time ASL Recognition</h1>
    <p>Using Lightweight NN & Farneback Optical Flow</p>
    
    <div class="video-container">
        <img src="{{ url_for('video_feed') }}" width="640" height="480">
    </div>
    
    <div class="prediction-box" id="prediction-text">
        Waiting for frames...
    </div>

    <script>
        // Poll for the latest prediction text every 300ms
        setInterval(() => {
            fetch('/get_prediction')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('prediction-text').innerText = data.prediction;
                });
        }, 300);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    from flask import url_for

    return render_template_string(HTML_TEMPLATE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/get_prediction")
def get_prediction():
    with prediction_lock:
        return jsonify({"prediction": latest_prediction})


if __name__ == "__main__":
    print("Starting Web Server. Open http://localhost:5001 in your browser.")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
