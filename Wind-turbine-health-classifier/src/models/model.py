# Model wrapper
"""
Wind Turbine Anomaly Detection — Model
=======================================
Model creation, training, evaluation, prediction, and main entry point.
Supports two models: Logistic Regression (baseline) and Random Forest.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    classification_report,
    precision_recall_curve,
    auc,
    log_loss,
)

from config import (
    LABEL_MAP,
    INVERSE_LABEL_MAP,
    TARGET_NAMES,
    STATUS_COLUMN,
    LR_CONFIG,
    LR_RETRAIN_CONFIG,
    RF_CONFIG,
    RF_RETRAIN_CONFIG,
)
from preprocessing import run_preprocessing_pipeline, scale_features


# ==============================================================================
# MODEL CREATION
# ==============================================================================

def create_logistic_regression(config=None):
    """Create a LogisticRegression with the given configuration."""
    if config is None:
        config = LR_CONFIG
    return LogisticRegression(**config)


def create_random_forest(config=None):
    """Create a RandomForestClassifier with the given configuration."""
    if config is None:
        config = RF_CONFIG
    return RandomForestClassifier(**config)


# ==============================================================================
# TRAINING
# ==============================================================================

def train_model(model, X_train, y_train):
    """Fit a model on the given training data."""
    model.fit(X_train, y_train)
    model_name = type(model).__name__
    print(f"{model_name} trained successfully.")
    if hasattr(model, "n_iter_"):
        print(f"Number of iterations used: {model.n_iter_[0]}")
    return model


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_model(model, X_val, y_val):
    """
    Compute and print evaluation metrics:
    - Macro F1-Score
    - Precision-Recall AUC
    - Log Loss
    - Full Classification Report
    Returns a dict of metrics.
    """
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    # Macro F1-Score
    f1_macro = f1_score(y_val, y_pred, average="macro")
    print(f"=== Macro F1-Score: {f1_macro:.4f} ===\n")

    # Precision-Recall AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    print(f"=== Precision-Recall AUC: {pr_auc:.4f} ===\n")

    # Log Loss
    loss = log_loss(y_val, y_proba)
    print(f"=== Log Loss: {loss:.4f} ===\n")

    # Classification Report
    print("=== Classification Report ===")
    print(classification_report(y_val, y_pred, target_names=TARGET_NAMES))

    return {
        "f1_macro": f1_macro,
        "pr_auc": pr_auc,
        "log_loss": loss,
    }


# ==============================================================================
# PREDICTION ON UNKNOWN DATA
# ==============================================================================

def predict_unknowns(model, X_unknown, df_unknown):
    """
    Generate predictions + probabilities on unknown data,
    map back to human-readable labels, and print distribution.
    """
    unknown_preds = model.predict(X_unknown)
    unknown_proba = model.predict_proba(X_unknown)[:, 1]

    df_unknown = df_unknown.copy()
    df_unknown["predicted_status"] = [INVERSE_LABEL_MAP[p] for p in unknown_preds]
    df_unknown["anomaly_probability"] = unknown_proba

    print(f"=== Predictions for {len(df_unknown)} Unknown timestamps ===")
    print(df_unknown["predicted_status"].value_counts())

    return df_unknown


def retrain_and_predict(model_factory, retrain_config, data, use_scaler=False):
    """
    Retrain model on 100% labeled data and predict unknowns.
    For Logistic Regression, set use_scaler=True.
    For Random Forest, set use_scaler=False (trees don't need scaling).
    """
    feature_cols = data["feature_cols"]
    df_labeled_sorted = data["df_labeled_sorted"]
    df_unknown = data["df_unknown"]

    X_full_labeled = df_labeled_sorted[feature_cols]
    y_full_labeled = df_labeled_sorted[STATUS_COLUMN].map(LABEL_MAP)
    X_unknown = data["X_unknown"]

    model_name = model_factory(retrain_config).__class__.__name__
    print(f"Training final {model_name} on 100% of labeled data...")

    if use_scaler:
        final_scaler = StandardScaler()
        X_full_scaled = final_scaler.fit_transform(X_full_labeled)
        X_unknown_input = final_scaler.transform(X_unknown)
    else:
        X_full_scaled = X_full_labeled
        X_unknown_input = X_unknown

    model = model_factory(retrain_config)
    model.fit(X_full_scaled, y_full_labeled)

    print(f"Predicting status for {len(X_unknown)} 'Unknown' winter timestamps...")

    y_pred_unknown = model.predict(X_unknown_input)
    y_proba_unknown = model.predict_proba(X_unknown_input)[:, 1]

    df_result = df_unknown.copy()
    df_result["predicted_status"] = pd.Series(y_pred_unknown).map(INVERSE_LABEL_MAP).values
    df_result["anomaly_probability"] = y_proba_unknown

    print(f"\n=== Final Prediction Distribution for 'Unknown' Timestamps ===")
    print(df_result["predicted_status"].value_counts(dropna=False))

    # Per-turbine breakdown
    print(f"\n=== Global Model — Predictions by Turbine ID ===")
    for aid, grp in df_result.groupby("asset_id"):
        n_normal = (grp["predicted_status"] == "Normal").sum()
        n_anom = (grp["predicted_status"] == "Anomalous").sum()
        print(f"  Turbine {aid}: Normal={n_normal}, Anomalous={n_anom} "
              f"(Anomaly rate: {n_anom/len(grp)*100:.1f}%)")

    return df_result


# ==============================================================================
# PIPELINE RUNNERS
# ==============================================================================

def run_logistic_regression_pipeline(data):
    """
    Full Logistic Regression pipeline:
    1. Scale features
    2. Train on 80% split
    3. Evaluate on 20% split
    4. Retrain on 100% labeled data and predict unknowns
    """
    print("\n" + "=" * 60)
    print("MODEL 1: LOGISTIC REGRESSION (BASELINE)")
    print("=" * 60)

    # Scale features (LR requires scaling)
    print("\n--- Feature Scaling ---")
    scale_result = scale_features(
        data["X_train"], data["X_val"], data["X_unknown"]
    )

    # Train
    print("\n--- Training ---")
    lr_model = create_logistic_regression()
    train_model(lr_model, scale_result["X_train_scaled"], data["y_train"])

    # Evaluate
    print("\n--- Evaluation ---")
    metrics = evaluate_model(lr_model, scale_result["X_val_scaled"], data["y_val"])

    # Predict unknowns (initial 80% model)
    print("\n--- Predicting Unknown Samples ---")
    predict_unknowns(lr_model, scale_result["X_unknown_scaled"], data["df_unknown"])

    # Retrain on 100% and final predictions
    print("\n--- Retraining on 100% Labeled Data ---")
    df_final = retrain_and_predict(
        create_logistic_regression, LR_RETRAIN_CONFIG, data, use_scaler=True
    )

    return metrics, df_final


def run_random_forest_pipeline(data):
    """
    Full Random Forest pipeline:
    1. Train on 80% split (no scaling needed — tree-based models are scale-invariant)
    2. Evaluate on 20% split
    3. Retrain on 100% labeled data and predict unknowns
    """
    print("\n" + "=" * 60)
    print("MODEL 2: RANDOM FOREST")
    print("=" * 60)

    # Train (no scaling needed for trees)
    print("\n--- Training ---")
    rf_model = create_random_forest()
    train_model(rf_model, data["X_train"], data["y_train"])

    # Evaluate
    print("\n--- Evaluation ---")
    metrics = evaluate_model(rf_model, data["X_val"], data["y_val"])

    # Predict unknowns (initial 80% model)
    print("\n--- Predicting Unknown Samples ---")
    predict_unknowns(rf_model, data["X_unknown"], data["df_unknown"])

    # Retrain on 100% and final predictions
    print("\n--- Retraining on 100% Labeled Data ---")
    df_final = retrain_and_predict(
        create_random_forest, RF_RETRAIN_CONFIG, data, use_scaler=False
    )

    return metrics, df_final


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Run the complete ML pipeline: preprocessing → LR → RF."""
    import warnings
    warnings.filterwarnings("ignore")

    # Step 1: Preprocessing
    data = run_preprocessing_pipeline()

    # Step 2: Logistic Regression
    lr_metrics, lr_predictions = run_logistic_regression_pipeline(data)

    # Step 3: Random Forest
    rf_metrics, rf_predictions = run_random_forest_pipeline(data)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Logistic Regression — F1-Macro: {lr_metrics['f1_macro']:.4f}, "
          f"PR-AUC: {lr_metrics['pr_auc']:.4f}")
    print(f"Random Forest       — F1-Macro: {rf_metrics['f1_macro']:.4f}, "
          f"PR-AUC: {rf_metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
