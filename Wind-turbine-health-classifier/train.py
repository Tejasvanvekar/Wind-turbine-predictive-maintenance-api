"""
Train and Save all models.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Ensure paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stuff')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from models.preprocessing import run_preprocessing_pipeline, scale_features
from models.model import (
    create_logistic_regression,
    create_random_forest,
    create_xgboost,
    train_model,
    evaluate_model,
)
from serialization import save_model

def main():
    print("Loading data...")
    data = run_preprocessing_pipeline()

    print("\n--- Training Logistic Regression ---")
    scale_result = scale_features(data["X_train"], data["X_val"])
    lr = create_logistic_regression()
    train_model(lr, scale_result["X_train_scaled"], data["y_train"])
    lr_metrics = evaluate_model(lr, scale_result["X_val_scaled"], data["y_val"])
    save_model(lr, "logistic_regression", lr_metrics, data["feature_cols"], scale_result["scaler"])

    print("\n--- Training Random Forest ---")
    rf = create_random_forest()
    train_model(rf, data["X_train"], data["y_train"])
    rf_metrics = evaluate_model(rf, data["X_val"], data["y_val"])
    save_model(rf, "random_forest", rf_metrics, data["feature_cols"], None)

    print("\n--- Training XGBoost ---")
    xgb = create_xgboost()
    train_model(xgb, data["X_train"], data["y_train"])
    xgb_metrics = evaluate_model(xgb, data["X_val"], data["y_val"])
    save_model(xgb, "xgboost", xgb_metrics, data["feature_cols"], None)

    print("\nAll models trained and serialized!")

if __name__ == "__main__":
    main()
