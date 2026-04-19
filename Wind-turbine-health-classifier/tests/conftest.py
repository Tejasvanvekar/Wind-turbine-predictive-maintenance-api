"""
Shared fixtures for all test modules.
Handles path setup so tests can import src/ and stuff/ modules.
"""

import os
import sys
import pytest


# ---------------------------------------------------------------------------
# Path setup: let tests import from src/ and stuff/ (pipeline modules)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
STUFF_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "stuff")

for p in (SRC_DIR, STUFF_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Lightweight synthetic dataset (used by unit tests — no disk I/O)
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd


@pytest.fixture
def synthetic_dataset():
    """
    Build a small synthetic dataset that mirrors the real pipeline's shape:
    - 200 samples, 10 numeric features, 1 timestamp, 1 status column
    - 80 % Normal, 15 % Anomalous, 5 % Unknown
    """
    rng = np.random.RandomState(42)
    n = 200

    timestamps = pd.date_range("2023-01-01", periods=n, freq="10min")
    features = {f"sensor_{i}": rng.randn(n) for i in range(10)}
    features["asset_id"] = rng.choice([38, 44], size=n)

    statuses = (
        ["Normal"] * 160
        + ["Anomalous"] * 30
        + ["Unknown"] * 10
    )
    rng.shuffle(statuses)

    df = pd.DataFrame(features)
    df["time_stamp"] = timestamps
    df["status"] = statuses
    return df


@pytest.fixture
def synthetic_train_val(synthetic_dataset):
    """
    Split the synthetic dataset the same way the real pipeline does:
    labeled / unknown → chronological 80/20 split → feature columns.
    Returns a dict matching the shape of run_preprocessing_pipeline().
    """
    df = synthetic_dataset
    drop_cols = ["time_stamp", "status"]
    label_map = {"Normal": 0, "Anomalous": 1}

    df_labeled = df[df["status"] != "Unknown"].copy()
    df_unknown = df[df["status"] == "Unknown"].copy()

    df_labeled = df_labeled.sort_values("time_stamp").reset_index(drop=True)
    feature_cols = [c for c in df_labeled.columns if c not in drop_cols]

    split_idx = int(len(df_labeled) * 0.8)

    X_all = df_labeled[feature_cols]
    y_all = df_labeled["status"].map(label_map)

    return {
        "X_train": X_all.iloc[:split_idx],
        "X_val": X_all.iloc[split_idx:],
        "y_train": y_all.iloc[:split_idx],
        "y_val": y_all.iloc[split_idx:],
        "X_unknown": df_unknown[feature_cols],
        "df_labeled_sorted": df_labeled,
        "df_unknown": df_unknown,
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# Loaded model artifact fixture (loads the real .joblib files)
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


@pytest.fixture(scope="session")
def rf_artifact():
    """Load the real random_forest .joblib once per session."""
    import joblib, glob
    files = sorted(glob.glob(os.path.join(MODELS_DIR, "random_forest_v*.joblib")))
    if not files:
        pytest.skip("No random_forest .joblib found — run serialization first")
    return joblib.load(files[-1])


@pytest.fixture(scope="session")
def lr_artifact():
    """Load the real logistic_regression .joblib once per session."""
    import joblib, glob
    files = sorted(glob.glob(os.path.join(MODELS_DIR, "logistic_regression_v*.joblib")))
    if not files:
        pytest.skip("No logistic_regression .joblib found — run serialization first")
    return joblib.load(files[-1])
