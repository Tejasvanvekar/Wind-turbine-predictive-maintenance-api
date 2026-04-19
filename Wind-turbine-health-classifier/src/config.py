# Configuration
"""
Wind Turbine Anomaly Detection — Configuration
================================================
Central configuration for the ML pipeline.
All hyperparameters, file paths, and constants are defined here.
"""

import os

# ==============================================================================
# FILE PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILES = [
    os.path.join(BASE_DIR, "wind_turbine_snippet_A.csv"),
    os.path.join(BASE_DIR, "wind_turbine_snippet_B.csv"),
]
CSV_SEPARATOR = ";"
TIMESTAMP_COLUMN = "time_stamp"

# ==============================================================================
# LABEL MAPPING
# ==============================================================================
# Machine learning models require numerical targets, not strings.
# Normal / Majority Class = 0
# Anomalous / Target Class = 1
LABEL_MAP = {"Normal": 0, "Anomalous": 1}
INVERSE_LABEL_MAP = {0: "Normal", 1: "Anomalous"}
TARGET_NAMES = ["Normal", "Anomalous"]

# Status values
STATUS_COLUMN = "status"
STATUS_UNKNOWN = "Unknown"

# Columns to drop from the feature space
DROP_COLUMNS = [TIMESTAMP_COLUMN, STATUS_COLUMN]

# ==============================================================================
# TRAIN / VALIDATION SPLIT
# ==============================================================================
# Chronological split ratio — first 80% for training, last 20% for validation.
# This prevents look-ahead bias (data leakage) in time-series data.
TRAIN_SPLIT_RATIO = 0.8
RANDOM_STATE = 42

# ==============================================================================
# LOGISTIC REGRESSION HYPERPARAMETERS
# ==============================================================================
LR_CONFIG = {
    "class_weight": {0: 1, 1: 3},   # Normal=1x weight, Anomalous=3x weight
    "C": 0.1,                        # L1 Penalty strength (strong regularization)
    "penalty": "l1",                 # Feature selection — shrinks useless weights to zero
    "solver": "liblinear",           # Required solver for L1 penalties
    "max_iter": 100,                 # Iterations for initial training
    "random_state": RANDOM_STATE,
}

# When retraining on 100% labeled data, allow more iterations for convergence
LR_RETRAIN_CONFIG = {
    **LR_CONFIG,
    "max_iter": 1000,
}

# ==============================================================================
# RANDOM FOREST HYPERPARAMETERS
# ==============================================================================
RF_CONFIG = {
    "n_estimators": 100,             # 100 trees for initial training
    "max_depth": 20,                 # Cap depth to prevent overfitting
    "class_weight": {0: 1, 1: 3},   # 3x penalty for missing anomalies
    "max_features": "sqrt",          # Force tree diversity
    "random_state": RANDOM_STATE,
    "n_jobs": -1,                    # Use all CPU cores
}

# When retraining on 100% labeled data, use more trees for stability
RF_RETRAIN_CONFIG = {
    **RF_CONFIG,
    "n_estimators": 200,             # 200 trees for maximum voting stability
}
