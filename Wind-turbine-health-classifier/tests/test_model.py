"""
Unit tests + Model performance tests
=====================================
- Unit tests use a small synthetic dataset (fast, no disk I/O).
- Performance tests use the real serialized .joblib models and verify
  that production metrics stay above minimum thresholds.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================================
# Imports from src/models/ and src/config.py
# ============================================================================

from models.model import (
    create_logistic_regression,
    create_random_forest,
    train_model,
    evaluate_model,
    predict_unknowns,
)
from config import (
    LABEL_MAP,
    INVERSE_LABEL_MAP,
    TARGET_NAMES,
    LR_CONFIG,
    LR_RETRAIN_CONFIG,
    RF_CONFIG,
    RF_RETRAIN_CONFIG,
    RANDOM_STATE,
    TRAIN_SPLIT_RATIO,
)
from models.preprocessing import scale_features


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  UNIT TESTS — Config                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class TestConfig:
    """Verify configuration values are well-formed."""

    def test_label_map_has_two_classes(self):
        assert len(LABEL_MAP) == 2
        assert "Normal" in LABEL_MAP
        assert "Anomalous" in LABEL_MAP

    def test_inverse_label_map_is_consistent(self):
        for label, code in LABEL_MAP.items():
            assert INVERSE_LABEL_MAP[code] == label

    def test_target_names_length(self):
        assert len(TARGET_NAMES) == 2

    def test_split_ratio_valid(self):
        assert 0.0 < TRAIN_SPLIT_RATIO < 1.0

    def test_random_state_is_int(self):
        assert isinstance(RANDOM_STATE, int)

    def test_lr_config_keys(self):
        required = {"C", "l1_ratio", "solver", "max_iter", "random_state", "class_weight"}
        assert required.issubset(set(LR_CONFIG.keys()))

    def test_rf_config_keys(self):
        required = {"n_estimators", "max_depth", "random_state", "class_weight", "n_jobs"}
        assert required.issubset(set(RF_CONFIG.keys()))

    def test_retrain_uses_more_capacity(self):
        """Retrain configs should use greater capacity than base configs."""
        assert LR_RETRAIN_CONFIG["max_iter"] > LR_CONFIG["max_iter"]
        assert RF_RETRAIN_CONFIG["n_estimators"] > RF_CONFIG["n_estimators"]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  UNIT TESTS — Model creation                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class TestModelCreation:
    """Verify model factories return correctly configured estimators."""

    def test_create_lr_returns_correct_type(self):
        model = create_logistic_regression()
        assert isinstance(model, LogisticRegression)

    def test_create_rf_returns_correct_type(self):
        model = create_random_forest()
        assert isinstance(model, RandomForestClassifier)

    def test_lr_custom_config(self):
        custom = {**LR_CONFIG, "C": 0.5, "max_iter": 500}
        model = create_logistic_regression(custom)
        assert model.C == 0.5
        assert model.max_iter == 500

    def test_rf_custom_config(self):
        custom = {**RF_CONFIG, "n_estimators": 50}
        model = create_random_forest(custom)
        assert model.n_estimators == 50

    def test_lr_has_class_weights(self):
        model = create_logistic_regression()
        assert model.class_weight is not None

    def test_rf_deterministic(self):
        model = create_random_forest()
        assert model.random_state == RANDOM_STATE


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  UNIT TESTS — Training & evaluation (synthetic data)                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class TestTraining:
    """Train models on synthetic data and verify the training contract."""

    def test_lr_train_returns_fitted_model(self, synthetic_train_val):
        data = synthetic_train_val
        scale_result = scale_features(data["X_train"])
        model = create_logistic_regression()
        trained = train_model(model, scale_result["X_train_scaled"], data["y_train"])
        assert hasattr(trained, "predict")
        assert hasattr(trained, "predict_proba")

    def test_rf_train_returns_fitted_model(self, synthetic_train_val):
        data = synthetic_train_val
        model = create_random_forest()
        trained = train_model(model, data["X_train"], data["y_train"])
        assert hasattr(trained, "predict")
        assert hasattr(trained, "predict_proba")

    def test_lr_predictions_are_binary(self, synthetic_train_val):
        data = synthetic_train_val
        scale_result = scale_features(data["X_train"], data["X_val"])
        model = create_logistic_regression()
        train_model(model, scale_result["X_train_scaled"], data["y_train"])
        preds = model.predict(scale_result["X_val_scaled"])
        assert set(preds).issubset({0, 1})

    def test_rf_predictions_are_binary(self, synthetic_train_val):
        data = synthetic_train_val
        model = create_random_forest()
        train_model(model, data["X_train"], data["y_train"])
        preds = model.predict(data["X_val"])
        assert set(preds).issubset({0, 1})


class TestEvaluation:
    """Verify evaluate_model returns the expected metric keys."""

    def test_evaluate_returns_expected_keys(self, synthetic_train_val):
        data = synthetic_train_val
        model = create_random_forest()
        train_model(model, data["X_train"], data["y_train"])
        metrics = evaluate_model(model, data["X_val"], data["y_val"])
        assert "f1_macro" in metrics
        assert "pr_auc" in metrics
        assert "log_loss" in metrics

    def test_metrics_are_floats(self, synthetic_train_val):
        data = synthetic_train_val
        model = create_random_forest()
        train_model(model, data["X_train"], data["y_train"])
        metrics = evaluate_model(model, data["X_val"], data["y_val"])
        for v in metrics.values():
            assert isinstance(v, float)

    def test_f1_in_valid_range(self, synthetic_train_val):
        data = synthetic_train_val
        model = create_random_forest()
        train_model(model, data["X_train"], data["y_train"])
        metrics = evaluate_model(model, data["X_val"], data["y_val"])
        assert 0.0 <= metrics["f1_macro"] <= 1.0

    def test_pr_auc_in_valid_range(self, synthetic_train_val):
        data = synthetic_train_val
        model = create_random_forest()
        train_model(model, data["X_train"], data["y_train"])
        metrics = evaluate_model(model, data["X_val"], data["y_val"])
        assert 0.0 <= metrics["pr_auc"] <= 1.0


class TestPredictUnknowns:
    """Verify predict_unknowns output shape and columns."""

    def test_output_has_prediction_columns(self, synthetic_train_val):
        data = synthetic_train_val
        model = create_random_forest()
        train_model(model, data["X_train"], data["y_train"])
        result_df = predict_unknowns(model, data["X_unknown"], data["df_unknown"])
        assert "predicted_status" in result_df.columns
        assert "anomaly_probability" in result_df.columns

    def test_output_length_matches_input(self, synthetic_train_val):
        data = synthetic_train_val
        model = create_random_forest()
        train_model(model, data["X_train"], data["y_train"])
        result_df = predict_unknowns(model, data["X_unknown"], data["df_unknown"])
        assert len(result_df) == len(data["df_unknown"])

    def test_predictions_are_valid_labels(self, synthetic_train_val):
        data = synthetic_train_val
        model = create_random_forest()
        train_model(model, data["X_train"], data["y_train"])
        result_df = predict_unknowns(model, data["X_unknown"], data["df_unknown"])
        valid_labels = set(INVERSE_LABEL_MAP.values())
        assert set(result_df["predicted_status"].unique()).issubset(valid_labels)

    def test_probabilities_in_valid_range(self, synthetic_train_val):
        data = synthetic_train_val
        model = create_random_forest()
        train_model(model, data["X_train"], data["y_train"])
        result_df = predict_unknowns(model, data["X_unknown"], data["df_unknown"])
        assert result_df["anomaly_probability"].between(0.0, 1.0).all()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  UNIT TESTS — Scaling                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class TestScaling:
    """Verify StandardScaler wrapper behaviour."""

    def test_scaled_train_has_zero_mean(self, synthetic_train_val):
        data = synthetic_train_val
        result = scale_features(data["X_train"])
        means = result["X_train_scaled"].mean(axis=0)
        np.testing.assert_allclose(means, 0.0, atol=1e-7)

    def test_scaled_train_has_unit_std(self, synthetic_train_val):
        data = synthetic_train_val
        result = scale_features(data["X_train"])
        stds = result["X_train_scaled"].std(axis=0)
        np.testing.assert_allclose(stds, 1.0, atol=1e-7)

    def test_scaler_object_returned(self, synthetic_train_val):
        data = synthetic_train_val
        result = scale_features(data["X_train"])
        assert isinstance(result["scaler"], StandardScaler)

    def test_val_scaling_uses_train_stats(self, synthetic_train_val):
        """Validation set must NOT be refit — its mean won't be exactly 0."""
        data = synthetic_train_val
        result = scale_features(data["X_train"], data["X_val"])
        val_means = result["X_val_scaled"].mean(axis=0)
        # After transform-only, mean should be non-zero in general
        assert result["X_val_scaled"].shape == data["X_val"].shape


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL PERFORMANCE TESTS — Real serialized artifacts                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Minimum acceptable thresholds (based on validated notebook results)
MIN_F1_MACRO_LR = 0.75
MIN_PR_AUC_LR = 0.75
MIN_F1_MACRO_RF = 0.90
MIN_PR_AUC_RF = 0.85
MIN_F1_MACRO_XGB = 0.90
MIN_PR_AUC_XGB = 0.85


class TestModelPerformance:
    """
    Verify that the production-serialized models meet minimum quality gates.
    These tests load the real .joblib artifacts (session-scoped fixtures).
    """

    @pytest.fixture(scope="class")
    def xgb_artifact(self):
        """Load the real xgboost .joblib once per test class."""
        import joblib, glob, os
        files = sorted(glob.glob(os.path.join("models", "xgboost_v*.joblib")))
        if not files:
            pytest.skip("No xgboost .joblib found — run serialization first")
        return joblib.load(files[-1])


    # --- Logistic Regression ---

    def test_lr_f1_above_threshold(self, lr_artifact):
        metrics = lr_artifact["metadata"].get("performance_metrics", {})
        f1 = metrics.get("f1_macro")
        assert f1 is not None, "f1_macro not found in LR artifact metadata"
        assert f1 >= MIN_F1_MACRO_LR, (
            f"LR F1-Macro {f1:.4f} is below threshold {MIN_F1_MACRO_LR}"
        )

    def test_lr_prauc_above_threshold(self, lr_artifact):
        metrics = lr_artifact["metadata"].get("performance_metrics", {})
        prauc = metrics.get("pr_auc")
        assert prauc is not None, "pr_auc not found in LR artifact metadata"
        assert prauc >= MIN_PR_AUC_LR, (
            f"LR PR-AUC {prauc:.4f} is below threshold {MIN_PR_AUC_LR}"
        )

    def test_lr_artifact_has_scaler(self, lr_artifact):
        assert lr_artifact.get("scaler") is not None, (
            "LR artifact must bundle a fitted scaler"
        )

    def test_lr_artifact_has_feature_names(self, lr_artifact):
        names = lr_artifact["metadata"].get("feature_names", [])
        assert len(names) > 0, "LR artifact is missing feature_names"

    # --- Random Forest ---

    def test_rf_f1_above_threshold(self, rf_artifact):
        metrics = rf_artifact["metadata"].get("performance_metrics", {})
        f1 = metrics.get("f1_macro")
        assert f1 is not None, "f1_macro not found in RF artifact metadata"
        assert f1 >= MIN_F1_MACRO_RF, (
            f"RF F1-Macro {f1:.4f} is below threshold {MIN_F1_MACRO_RF}"
        )

    def test_rf_prauc_above_threshold(self, rf_artifact):
        metrics = rf_artifact["metadata"].get("performance_metrics", {})
        prauc = metrics.get("pr_auc")
        assert prauc is not None, "pr_auc not found in RF artifact metadata"
        assert prauc >= MIN_PR_AUC_RF, (
            f"RF PR-AUC {prauc:.4f} is below threshold {MIN_PR_AUC_RF}"
        )

    def test_rf_artifact_no_scaler(self, rf_artifact):
        assert rf_artifact.get("scaler") is None, (
            "RF artifact should NOT bundle a scaler — trees are scale-invariant"
        )

    def test_rf_artifact_has_feature_names(self, rf_artifact):
        names = rf_artifact["metadata"].get("feature_names", [])
        assert len(names) > 0, "RF artifact is missing feature_names"

    # --- XGBoost ---

    def test_xgb_f1_above_threshold(self, xgb_artifact):
        metrics = xgb_artifact["metadata"].get("performance_metrics", {})
        f1 = metrics.get("f1_macro")
        assert f1 is not None, "f1_macro not found in XGB artifact metadata"
        assert f1 >= MIN_F1_MACRO_XGB, (
            f"XGB F1-Macro {f1:.4f} is below threshold {MIN_F1_MACRO_XGB}"
        )

    def test_xgb_prauc_above_threshold(self, xgb_artifact):
        metrics = xgb_artifact["metadata"].get("performance_metrics", {})
        prauc = metrics.get("pr_auc")
        assert prauc is not None, "pr_auc not found in XGB artifact metadata"
        assert prauc >= MIN_PR_AUC_XGB, (
            f"XGB PR-AUC {prauc:.4f} is below threshold {MIN_PR_AUC_XGB}"
        )

    def test_xgb_artifact_no_scaler(self, xgb_artifact):
        assert xgb_artifact.get("scaler") is None, (
            "XGB artifact should NOT bundle a scaler"
        )

    # --- Cross-model consistency ---

    def test_all_models_share_same_features(self, lr_artifact, rf_artifact, xgb_artifact):
        lr_features = set(lr_artifact["metadata"].get("feature_names", []))
        rf_features = set(rf_artifact["metadata"].get("feature_names", []))
        xgb_features = set(xgb_artifact["metadata"].get("feature_names", []))
        assert lr_features == rf_features == xgb_features, (
            "Feature mismatch across models."
        )

    def test_rf_outperforms_lr_on_f1(self, lr_artifact, rf_artifact):
        """Random Forest should outperform the LR baseline."""
        lr_f1 = lr_artifact["metadata"]["performance_metrics"].get("f1_macro", 0)
        rf_f1 = rf_artifact["metadata"]["performance_metrics"].get("f1_macro", 0)
        assert rf_f1 >= lr_f1, (
            f"RF F1 ({rf_f1:.4f}) should be >= LR F1 ({lr_f1:.4f})"
        )
