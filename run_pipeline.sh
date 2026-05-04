#!/bin/bash
echo "1. Extracting Farneback Optical Flow features..."
python -m src.extract_features

echo "2. Training Lightweight NN on Flow Features..."
python -m src.model

echo "Pipeline complete. You can now run the realtime demo:"
echo "python -m src.realtime_demo"
