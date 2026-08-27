#!/usr/bin/env python3
"""
Run the complete DGA research pipeline offline.

Usage:
    python run_full_experiment.py
    python run_full_experiment.py --use-snorkel --seed 42

This script intentionally does NOT run inside the production web request path.
"""
from train_unsupervised_models import main

if __name__ == "__main__":
    main()