#!/usr/bin/env bash
set -e

python -m pip install -r requirements.txt
python train_transformer_degradation.py

echo "Training complete. Run 'python api.py' to start the Flask API."