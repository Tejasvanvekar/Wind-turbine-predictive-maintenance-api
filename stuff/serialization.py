"""
Wind Turbine Anomaly Detection — Model Serialization
=====================================================
Save, load, and inspect trained models with versioning and metadata.
Uses joblib (recommended by scikit-learn for models containing large numpy arrays).
"""

import os
import glob
import re
from datetime import datetime, timezone

import joblib
import sklearn

from config import MODELS_DIR, MODEL_FILE_EXTENSION


# ==============================================================================
# VERSIONING HELPERS
# ==============================================================================

def _get_next_version(model_type):
    """
    Scan the models directory for existing files matching the pattern
    `{model_type}_v{N}.joblib` and return the next version number.
    """
    pattern = os.path.join(
        MODELS_DIR, f"{model_type}_v*{MODEL_FILE_EXTENSION}"
    )
    existing = glob.glob(pattern)

    if not existing:
        return 1

    # Extract version numbers from filenames
    versions = []
    for filepath in existing:
        basename = os.path.basename(filepath)
        match = re.search(r"_v(\d+)", basename)
        if match:
            versions.append(int(match.group(1)))

    return max(versions) + 1 if versions else 1


def _build_filepath(model_type, version):
    """Build the full filepath for a versioned model."""
    filename = f"{model_type}_v{version}{MODEL_FILE_EXTENSION}"
    return os.path.join(MODELS_DIR, filename)


# ==============================================================================
# SAVE
# ==============================================================================

def save_model(model, model_type, metrics=None, feature_cols=None, scaler=None):
    """
    Serialize a trained model to disk with metadata.

    The saved artifact is a dict containing:
        - model: the trained sklearn estimator
        - scaler: the fitted StandardScaler (if provided, e.g. for LR)
        - metadata: training date, version, metrics, hyperparameters, etc.

    Parameters
    ----------
    model : sklearn estimator
        A fitted model (LogisticRegression, RandomForestClassifier, etc.)
    model_type : str
        Identifier used for filename, e.g. "logistic_regression" or "random_forest"
    metrics : dict, optional
        Performance metrics from evaluate_model() (f1_macro, pr_auc, log_loss)
    feature_cols : list[str], optional
        Feature column names used during training
    scaler : StandardScaler, optional
        Fitted scaler to bundle with the model (needed for LR inference)

    Returns
    -------
    str
        The filepath where the model was saved.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    version = _get_next_version(model_type)
    filepath = _build_filepath(model_type, version)

    artifact = {
        "model": model,
        "scaler": scaler,
        "metadata": {
            "model_type": model_type,
            "model_version": version,
            "training_date": datetime.now(timezone.utc).isoformat(),
            "performance_metrics": metrics or {},
            "hyperparameters": model.get_params(),
            "feature_names": list(feature_cols) if feature_cols is not None else [],
            "sklearn_version": sklearn.__version__,
            "model_class": type(model).__name__,
        },
    }

    joblib.dump(artifact, filepath)

    print(f"\n{'='*60}")
    print(f"MODEL SAVED")
    print(f"{'='*60}")
    print(f"  File:    {filepath}")
    print(f"  Type:    {model_type}")
    print(f"  Version: v{version}")
    if metrics:
        print(f"  F1-Macro: {metrics.get('f1_macro', 'N/A'):.4f}")
        print(f"  PR-AUC:   {metrics.get('pr_auc', 'N/A'):.4f}")
        print(f"  Log-Loss: {metrics.get('log_loss', 'N/A'):.4f}")
    print(f"  Scaler:  {'included' if scaler is not None else 'none'}")

    return filepath


# ==============================================================================
# LOAD
# ==============================================================================

def load_model(filepath):
    """
    Load a serialized model artifact from disk.

    Parameters
    ----------
    filepath : str
        Path to a .joblib model file.

    Returns
    -------
    dict
        The full artifact dict with keys: 'model', 'scaler', 'metadata'.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file does not contain expected keys.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    artifact = joblib.load(filepath)

    # Validate structure
    expected_keys = {"model", "metadata"}
    if not expected_keys.issubset(artifact.keys()):
        raise ValueError(
            f"Invalid model file. Expected keys {expected_keys}, "
            f"got {set(artifact.keys())}"
        )

    meta = artifact["metadata"]
    print(f"Loaded {meta['model_class']} v{meta['model_version']} "
          f"(trained {meta['training_date']})")

    return artifact


# ==============================================================================
# INSPECT
# ==============================================================================

def get_model_info(filepath):
    """
    Load and display the metadata of a saved model (model card).

    Parameters
    ----------
    filepath : str
        Path to a .joblib model file.

    Returns
    -------
    dict
        The metadata dict.
    """
    artifact = load_model(filepath)
    meta = artifact["metadata"]

    print(f"\n{'='*60}")
    print(f"MODEL CARD")
    print(f"{'='*60}")
    print(f"  Model Class:     {meta['model_class']}")
    print(f"  Model Type:      {meta['model_type']}")
    print(f"  Version:         v{meta['model_version']}")
    print(f"  Training Date:   {meta['training_date']}")
    print(f"  Sklearn Version: {meta['sklearn_version']}")
    print(f"  Scaler Bundled:  {'yes' if artifact.get('scaler') is not None else 'no'}")

    if meta.get("performance_metrics"):
        print(f"\n  --- Performance Metrics ---")
        for key, val in meta["performance_metrics"].items():
            print(f"  {key}: {val:.4f}")

    if meta.get("feature_names"):
        print(f"\n  --- Features ({len(meta['feature_names'])}) ---")
        for feat in meta["feature_names"]:
            print(f"    • {feat}")

    if meta.get("hyperparameters"):
        print(f"\n  --- Hyperparameters ---")
        for key, val in meta["hyperparameters"].items():
            print(f"  {key}: {val}")

    return meta
