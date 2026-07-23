#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/backend"

python -m pip install -r requirements.txt
python train_models.py

echo "Training complete. Run 'python app.py' (from backend/) to start the Flask API."
