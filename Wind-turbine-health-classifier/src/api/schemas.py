"""
Wind Turbine Health Classifier — API Schemas
=============================================
Pydantic models for request/response validation and structured errors.
"""

import math
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# ==============================================================================
# ENUMS
# ==============================================================================

class ModelType(str, Enum):
    """Supported model types."""
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


# ==============================================================================
# REQUEST SCHEMAS
# ==============================================================================

# Maximum number of samples in a single batch request
BATCH_MAX_SAMPLES = 1000


class PredictionRequest(BaseModel):
    """
    Input payload for the POST /predict endpoint.

    The `features` dict maps feature names to their float values.
    All features the model was trained with must be provided.
    `model_type` selects which loaded model to use (defaults to random_forest).

    Example::

        {
            "features": {
                "Wind speed 1 (avg)": 5.2,
                "Rotor speed 1 (avg)": 12.3,
                ...
            },
            "model_type": "random_forest"
        }
    """
    features: dict[str, float] = Field(
        ...,
        description="Dict of feature name → float value. Must include all features the model expects.",
        min_length=1,
    )
    model_type: ModelType = Field(
        default=ModelType.RANDOM_FOREST,
        description="Model to use: 'logistic_regression' or 'random_forest'",
    )

    @field_validator("features")
    @classmethod
    def validate_feature_values(cls, v: dict[str, float]) -> dict[str, float]:
        """Reject NaN, Inf, and -Inf feature values."""
        invalid = {
            name: val for name, val in v.items()
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val)
        }
        if invalid:
            raise ValueError(
                f"Feature values must be finite numbers. "
                f"Invalid features: {invalid}"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": {
                        "Wind speed 1 (avg)": 5.2,
                        "Rotor speed 1 (avg)": 12.3,
                        "Nacelle outside temperature (avg)": 8.1,
                    },
                    "model_type": "random_forest",
                }
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """
    Input payload for the POST /batch-predict endpoint.

    `samples` is a list of feature dicts — one per turbine observation.
    All samples must contain the same features the model was trained with.
    Maximum batch size: 1000 samples.

    Example::

        {
            "samples": [
                {"Wind speed 1 (avg)": 5.2, "Rotor speed 1 (avg)": 12.3, ...},
                {"Wind speed 1 (avg)": 3.1, "Rotor speed 1 (avg)": 8.7, ...}
            ],
            "model_type": "random_forest"
        }
    """
    samples: list[dict[str, float]] = Field(
        ...,
        description="List of feature dicts — one per observation.",
        min_length=1,
        max_length=BATCH_MAX_SAMPLES,
    )
    model_type: ModelType = Field(
        default=ModelType.RANDOM_FOREST,
        description="Model to use: 'logistic_regression' or 'random_forest'",
    )

    @field_validator("samples")
    @classmethod
    def validate_all_samples(cls, v: list[dict[str, float]]) -> list[dict[str, float]]:
        """Reject NaN/Inf values in any sample."""
        for i, sample in enumerate(v):
            if not sample:
                raise ValueError(f"Sample at index {i} is empty.")
            invalid = {
                name: val for name, val in sample.items()
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val)
            }
            if invalid:
                raise ValueError(
                    f"Sample at index {i} contains invalid values "
                    f"(NaN or Inf not allowed): {invalid}"
                )
        return v


# ==============================================================================
# RESPONSE SCHEMAS
# ==============================================================================

class PredictionResponse(BaseModel):
    """Response from the POST /predict endpoint."""
    prediction: str = Field(
        ...,
        description="Predicted class: 'Normal' or 'Anomalous'",
    )
    anomaly_probability: float = Field(
        ...,
        description="Probability of the 'Anomalous' class (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    model_type: str = Field(
        ...,
        description="Model type used for prediction",
    )
    model_version: int = Field(
        ...,
        description="Version of the model used",
    )


class BatchPredictionItem(BaseModel):
    """A single prediction within a batch response."""
    prediction: str
    anomaly_probability: float = Field(ge=0.0, le=1.0)


class BatchPredictionResponse(BaseModel):
    """Response from the POST /batch-predict endpoint."""
    predictions: list[BatchPredictionItem] = Field(
        ...,
        description="List of predictions, one per input sample (same order)",
    )
    model_type: str
    model_version: int
    total_samples: int


class HealthResponse(BaseModel):
    """Response from the GET /health endpoint."""
    status: str = Field(
        ...,
        description="API health status: 'healthy' or 'degraded'",
    )
    loaded_models: list[str] = Field(
        default_factory=list,
        description="List of model types currently loaded in memory",
    )
    models_dir: str = Field(
        ...,
        description="Path to the models directory",
    )


class ModelInfoResponse(BaseModel):
    """Response from the GET /model-info endpoint."""
    model_type: str
    model_class: str
    model_version: int
    training_date: str
    sklearn_version: str
    has_scaler: bool
    performance_metrics: dict = Field(default_factory=dict)
    hyperparameters: dict = Field(default_factory=dict)
    feature_names: list[str] = Field(default_factory=list)
    feature_count: int = 0


# ==============================================================================
# ERROR RESPONSE SCHEMA
# ==============================================================================

class ErrorDetail(BaseModel):
    """Structured error response returned by all error handlers."""
    error_code: str = Field(
        ...,
        description="Machine-readable error code (e.g. 'MODEL_NOT_FOUND')",
    )
    message: str = Field(
        ...,
        description="Human-readable error description",
    )
    details: dict | list | None = Field(
        default=None,
        description="Additional context (missing features, sample index, etc.)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error_code": "MISSING_FEATURES",
                    "message": "Request is missing 2 required features.",
                    "details": {
                        "missing_features": ["Wind speed 1 (avg)", "Rotor speed 1 (avg)"],
                        "expected_count": 200,
                        "provided_count": 198,
                    },
                }
            ]
        }
    }
