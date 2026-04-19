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
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
MODEL_FILE_EXTENSION = ".joblib"

PROJECT_ROOT = os.path.dirname(BASE_DIR)
PRODUCTION_ROOT = os.path.dirname(PROJECT_ROOT)
STUFF_DIR = os.path.join(PRODUCTION_ROOT, "stuff")

DATA_FILES = [
    os.path.join(STUFF_DIR, "wind_turbine_snippet_A.csv"),
    os.path.join(STUFF_DIR, "wind_turbine_snippet_B.csv"),
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
    "class_weight": {0: 1, 1: 3},
    "C": float(os.getenv("LR_C", "0.1")),
    "l1_ratio": float(os.getenv("LR_L1_RATIO", "1.0")),
    "solver": os.getenv("LR_SOLVER", "liblinear"),
    "max_iter": int(os.getenv("LR_MAX_ITER", "100")),
    "random_state": RANDOM_STATE,
}

LR_RETRAIN_CONFIG = {
    **LR_CONFIG,
    "max_iter": int(os.getenv("LR_RETRAIN_MAX_ITER", "1000")),
}

# ==============================================================================
# RANDOM FOREST HYPERPARAMETERS
# ==============================================================================
RF_CONFIG = {
    "n_estimators": int(os.getenv("RF_N_ESTIMATORS", "100")),
    "max_depth": int(os.getenv("RF_MAX_DEPTH", "20")),
    "class_weight": {0: 1, 1: 3},
    "max_features": os.getenv("RF_MAX_FEATURES", "sqrt"),
    "random_state": RANDOM_STATE,
    "n_jobs": int(os.getenv("RF_N_JOBS", "-1")),
}

RF_RETRAIN_CONFIG = {
    **RF_CONFIG,
    "n_estimators": int(os.getenv("RF_RETRAIN_N_ESTIMATORS", "200")),
}

# ==============================================================================
# XGBOOST HYPERPARAMETERS
# ==============================================================================
XGB_CONFIG = {
    "n_estimators": int(os.getenv("XGB_N_ESTIMATORS", "400")),
    "max_depth": int(os.getenv("XGB_MAX_DEPTH", "7")),
    "learning_rate": float(os.getenv("XGB_LEARNING_RATE", "0.02146")),
    "subsample": float(os.getenv("XGB_SUBSAMPLE", "0.8953")),
    "colsample_bytree": float(os.getenv("XGB_COLSAMPLE", "0.6213")),
    "min_child_weight": int(os.getenv("XGB_MIN_CHILD_WEIGHT", "4")),
    "gamma": float(os.getenv("XGB_GAMMA", "0.8409")),
    "reg_alpha": float(os.getenv("XGB_REG_ALPHA", "0.1854")),
    "reg_lambda": float(os.getenv("XGB_REG_LAMBDA", "8.096e-08")),
    "scale_pos_weight": float(os.getenv("XGB_SCALE_POS_WEIGHT", "7.25")),
    "random_state": RANDOM_STATE,
    "n_jobs": int(os.getenv("XGB_N_JOBS", "-1")),
    "eval_metric": "logloss",
}

XGB_RETRAIN_CONFIG = {
    **XGB_CONFIG,
    "n_estimators": int(os.getenv("XGB_RETRAIN_N_ESTIMATORS", "200")),
}

