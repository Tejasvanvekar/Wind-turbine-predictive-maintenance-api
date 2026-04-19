"""
Wind Turbine Health Classifier — API Routes
============================================
REST endpoints: /health, /model-info, /predict, /batch-predict
All routes use structured error responses and proper logging.
"""

import logging
import time

import numpy as np
from fastapi import APIRouter, HTTPException

from .schemas import (
    ModelType,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchPredictionItem,
    HealthResponse,
    ModelInfoResponse,
)

logger = logging.getLogger("wthc.api.routes")

router = APIRouter()

# Label mapping (matches training config)
LABEL_MAP = {0: "Normal", 1: "Anomalous"}


# ==============================================================================
# HELPERS
# ==============================================================================

def _get_artifact(model_type: ModelType):
    """
    Look up a loaded model artifact by type.
    Raises HTTPException 404 with structured error if not found.
    """
    from .main import loaded_models

    model_key = model_type.value

    if model_key not in loaded_models:
        available = list(loaded_models.keys()) or ["none"]
        logger.warning(
            "Model lookup failed: type=%s, available=%s", model_key, available
        )
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "MODEL_NOT_FOUND",
                "message": f"Model type '{model_key}' is not loaded.",
                "details": {"available_models": available},
            },
        )
    return loaded_models[model_key]


def _validate_features(features: dict[str, float], expected: list[str]):
    """
    Validate that the provided features exactly match the expected set.
    Raises HTTPException 422 with structured detail on mismatch.
    """
    if not expected:
        return

    provided = set(features.keys())
    expected_set = set(expected)
    missing = expected_set - provided
    extra = provided - expected_set

    if missing:
        logger.warning(
            "Feature validation failed: %d missing features", len(missing)
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "MISSING_FEATURES",
                "message": f"Request is missing {len(missing)} required feature(s).",
                "details": {
                    "missing_features": sorted(missing),
                    "expected_count": len(expected_set),
                    "provided_count": len(provided),
                },
            },
        )

    if extra:
        logger.warning(
            "Feature validation failed: %d unexpected features", len(extra)
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "UNEXPECTED_FEATURES",
                "message": f"Request contains {len(extra)} unexpected feature(s).",
                "details": {
                    "extra_features": sorted(extra),
                    "expected_count": len(expected_set),
                },
            },
        )


def _build_feature_array(features: dict[str, float], expected: list[str]):
    """Build a numpy row vector with features in the correct column order."""
    if expected:
        values = [features[f] for f in expected]
    else:
        values = list(features.values())
    return np.array(values).reshape(1, -1)


# ==============================================================================
# GET /health
# ==============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health_check():
    """
    Returns the API health status and which models are currently loaded.
    Status is **healthy** if at least one model is loaded, otherwise **degraded**.
    """
    from .main import loaded_models, MODELS_DIR

    status = "healthy" if loaded_models else "degraded"
    logger.info("Health check: status=%s, models=%d", status, len(loaded_models))

    return HealthResponse(
        status=status,
        loaded_models=list(loaded_models.keys()),
        models_dir=MODELS_DIR,
    )


# ==============================================================================
# GET /model-info
# ==============================================================================

@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Get model metadata",
    tags=["Models"],
)
async def model_info(model_type: ModelType = ModelType.RANDOM_FOREST):
    """
    Returns the full model card (metadata, performance metrics,
    hyperparameters, feature list) for the given model type.

    **Query parameter** `model_type`: `logistic_regression` or `random_forest`
    (defaults to `random_forest`).
    """
    artifact = _get_artifact(model_type)
    meta = artifact["metadata"]

    logger.info(
        "Model info requested: type=%s, version=v%s",
        meta["model_type"], meta["model_version"]
    )

    return ModelInfoResponse(
        model_type=meta["model_type"],
        model_class=meta["model_class"],
        model_version=meta["model_version"],
        training_date=meta["training_date"],
        sklearn_version=meta["sklearn_version"],
        has_scaler=artifact.get("scaler") is not None,
        performance_metrics=meta.get("performance_metrics", {}),
        hyperparameters=meta.get("hyperparameters", {}),
        feature_names=meta.get("feature_names", []),
        feature_count=len(meta.get("feature_names", [])),
    )


# ==============================================================================
# POST /predict
# ==============================================================================

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Single prediction",
    tags=["Prediction"],
)
async def predict(request: PredictionRequest):
    """
    Run anomaly prediction for a single turbine observation.

    The request body must contain **all features** the model was trained with.
    Features are passed as a `{name: value}` dict.
    Optionally specify `model_type` (defaults to `random_forest`).
    """
    start = time.perf_counter()

    artifact = _get_artifact(request.model_type)
    model = artifact["model"]
    scaler = artifact.get("scaler")
    meta = artifact["metadata"]
    expected_features = meta.get("feature_names", [])

    # Validate features
    _validate_features(request.features, expected_features)

    # Build feature array in correct column order
    X = _build_feature_array(request.features, expected_features)

    # Apply scaler if bundled (e.g. for Logistic Regression)
    if scaler is not None:
        X = scaler.transform(X)

    # Predict
    prediction_int = model.predict(X)[0]
    anomaly_proba = model.predict_proba(X)[0, 1]

    label = LABEL_MAP.get(prediction_int, str(prediction_int))
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Prediction: model=%s v%s, result=%s, prob=%.4f, time=%.1fms",
        request.model_type.value, meta["model_version"],
        label, anomaly_proba, elapsed_ms
    )

    return PredictionResponse(
        prediction=label,
        anomaly_probability=round(float(anomaly_proba), 6),
        model_type=request.model_type.value,
        model_version=meta["model_version"],
    )


# ==============================================================================
# POST /batch-predict
# ==============================================================================

@router.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    summary="Batch prediction",
    tags=["Prediction"],
)
async def batch_predict(request: BatchPredictionRequest):
    """
    Run anomaly prediction for **multiple** turbine observations in one call.

    Each sample in `samples` must contain all features the model was trained with.
    Results are returned in the same order as the input samples.
    Optionally specify `model_type` (defaults to `random_forest`).
    Maximum batch size: 1000 samples.
    """
    start = time.perf_counter()

    artifact = _get_artifact(request.model_type)
    model = artifact["model"]
    scaler = artifact.get("scaler")
    meta = artifact["metadata"]
    expected_features = meta.get("feature_names", [])

    # Validate every sample
    for i, sample in enumerate(request.samples):
        try:
            _validate_features(sample, expected_features)
        except HTTPException as e:
            # Add sample index to the error detail for debugging
            detail = e.detail
            if isinstance(detail, dict):
                detail["details"]["sample_index"] = i
            raise HTTPException(status_code=e.status_code, detail=detail)

    # Build feature matrix (N samples × M features)
    if expected_features:
        rows = [[sample[f] for f in expected_features] for sample in request.samples]
    else:
        rows = [list(sample.values()) for sample in request.samples]

    X = np.array(rows)

    # Apply scaler if bundled
    if scaler is not None:
        X = scaler.transform(X)

    # Predict all at once (efficient — single sklearn call)
    predictions_int = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Build response
    items = [
        BatchPredictionItem(
            prediction=LABEL_MAP.get(int(pred), str(pred)),
            anomaly_probability=round(float(prob), 6),
        )
        for pred, prob in zip(predictions_int, probabilities)
    ]

    elapsed_ms = (time.perf_counter() - start) * 1000
    n_anomalous = sum(1 for item in items if item.prediction == "Anomalous")

    logger.info(
        "Batch prediction: model=%s v%s, samples=%d, anomalous=%d, time=%.1fms",
        request.model_type.value, meta["model_version"],
        len(items), n_anomalous, elapsed_ms
    )

    return BatchPredictionResponse(
        predictions=items,
        model_type=request.model_type.value,
        model_version=meta["model_version"],
        total_samples=len(items),
    )
