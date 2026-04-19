"""
Integration tests
==================
End-to-end tests that verify the FastAPI application works correctly:
- Health endpoint
- Model info endpoint
- Single prediction
- Batch prediction
- Error handling (missing model, missing features, NaN values)

Uses the FastAPI TestClient (synchronous) so no real server is needed.
"""

import math
import pytest
from fastapi.testclient import TestClient


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def client():
    """
    Create a FastAPI TestClient.
    The app's lifespan loads models from the models/ directory automatically.
    """
    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_features(client):
    """
    Fetch the feature list from the model-info endpoint and build
    a valid sample dict (all features set to 1.0).
    """
    resp = client.get("/model-info", params={"model_type": "random_forest"})
    if resp.status_code != 200:
        pytest.skip("No model loaded — cannot build sample features")

    feature_names = resp.json().get("feature_names", [])
    if not feature_names:
        pytest.skip("Model has no feature_names in metadata")

    return {name: 1.0 for name in feature_names}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HEALTH ENDPOINT                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class TestHealthEndpoint:
    """GET /health — system health and loaded model status."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, client):
        body = client.get("/health").json()
        assert "status" in body
        assert body["status"] in ("healthy", "degraded")

    def test_health_lists_loaded_models(self, client):
        body = client.get("/health").json()
        assert "loaded_models" in body
        assert isinstance(body["loaded_models"], list)

    def test_health_includes_models_dir(self, client):
        body = client.get("/health").json()
        assert "models_dir" in body


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL INFO ENDPOINT                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class TestModelInfoEndpoint:
    """GET /model-info — model card retrieval."""

    def test_rf_info_returns_200(self, client):
        resp = client.get("/model-info", params={"model_type": "random_forest"})
        assert resp.status_code == 200

    def test_lr_info_returns_200(self, client):
        resp = client.get("/model-info", params={"model_type": "logistic_regression"})
        assert resp.status_code == 200

    def test_info_has_required_fields(self, client):
        body = client.get("/model-info", params={"model_type": "random_forest"}).json()
        for key in ("model_type", "model_class", "model_version",
                     "training_date", "feature_count", "performance_metrics"):
            assert key in body, f"Missing key: {key}"

    def test_feature_count_positive(self, client):
        body = client.get("/model-info", params={"model_type": "random_forest"}).json()
        assert body["feature_count"] > 0

    def test_invalid_model_type_returns_422(self, client):
        resp = client.get("/model-info", params={"model_type": "xgboost"})
        assert resp.status_code == 422


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SINGLE PREDICTION ENDPOINT                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class TestPredictEndpoint:
    """POST /predict — single observation prediction."""

    def test_predict_returns_200(self, client, sample_features):
        resp = client.post("/predict", json={
            "features": sample_features,
            "model_type": "random_forest",
        })
        assert resp.status_code == 200

    def test_predict_response_has_required_fields(self, client, sample_features):
        body = client.post("/predict", json={
            "features": sample_features,
            "model_type": "random_forest",
        }).json()
        for key in ("prediction", "anomaly_probability", "model_type", "model_version"):
            assert key in body, f"Missing key: {key}"

    def test_prediction_is_valid_label(self, client, sample_features):
        body = client.post("/predict", json={
            "features": sample_features,
            "model_type": "random_forest",
        }).json()
        assert body["prediction"] in ("Normal", "Anomalous")

    def test_probability_in_valid_range(self, client, sample_features):
        body = client.post("/predict", json={
            "features": sample_features,
            "model_type": "random_forest",
        }).json()
        p = body["anomaly_probability"]
        assert 0.0 <= p <= 1.0

    def test_lr_prediction_also_works(self, client, sample_features):
        resp = client.post("/predict", json={
            "features": sample_features,
            "model_type": "logistic_regression",
        })
        assert resp.status_code == 200
        assert resp.json()["prediction"] in ("Normal", "Anomalous")

    def test_both_models_return_same_feature_count(self, client, sample_features):
        """Both models should accept the same feature set."""
        for mt in ("random_forest", "logistic_regression"):
            resp = client.post("/predict", json={
                "features": sample_features,
                "model_type": mt,
            })
            assert resp.status_code == 200


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  BATCH PREDICTION ENDPOINT                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class TestBatchPredictEndpoint:
    """POST /batch-predict — multi-observation prediction."""

    def test_batch_returns_200(self, client, sample_features):
        resp = client.post("/batch-predict", json={
            "samples": [sample_features, sample_features],
            "model_type": "random_forest",
        })
        assert resp.status_code == 200

    def test_batch_response_count_matches(self, client, sample_features):
        n = 5
        resp = client.post("/batch-predict", json={
            "samples": [sample_features] * n,
            "model_type": "random_forest",
        })
        body = resp.json()
        assert body["total_samples"] == n
        assert len(body["predictions"]) == n

    def test_batch_predictions_are_valid(self, client, sample_features):
        resp = client.post("/batch-predict", json={
            "samples": [sample_features, sample_features],
            "model_type": "random_forest",
        })
        for item in resp.json()["predictions"]:
            assert item["prediction"] in ("Normal", "Anomalous")
            assert 0.0 <= item["anomaly_probability"] <= 1.0

    def test_batch_identical_inputs_give_identical_outputs(self, client, sample_features):
        resp = client.post("/batch-predict", json={
            "samples": [sample_features, sample_features],
            "model_type": "random_forest",
        })
        preds = resp.json()["predictions"]
        assert preds[0] == preds[1], "Identical inputs should give identical outputs"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ERROR HANDLING                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class TestErrorHandling:
    """Verify the API returns structured errors for invalid input."""

    def test_missing_features_returns_422(self, client):
        """Sending fewer features than expected should fail."""
        resp = client.post("/predict", json={
            "features": {"fake_feature": 1.0},
            "model_type": "random_forest",
        })
        assert resp.status_code == 422

    def test_empty_features_returns_422(self, client):
        resp = client.post("/predict", json={
            "features": {},
            "model_type": "random_forest",
        })
        assert resp.status_code == 422

    def test_nan_feature_rejected(self, client, sample_features):
        """NaN is not valid JSON — the client should raise ValueError."""
        bad = {**sample_features}
        first_key = next(iter(bad))
        bad[first_key] = float("nan")
        with pytest.raises(ValueError):
            client.post("/predict", json={
                "features": bad,
                "model_type": "random_forest",
            })

    def test_inf_feature_rejected(self, client, sample_features):
        """Inf is not valid JSON — the client should raise ValueError."""
        bad = {**sample_features}
        first_key = next(iter(bad))
        bad[first_key] = float("inf")
        with pytest.raises(ValueError):
            client.post("/predict", json={
                "features": bad,
                "model_type": "random_forest",
            })

    def test_empty_batch_returns_422(self, client):
        resp = client.post("/batch-predict", json={
            "samples": [],
            "model_type": "random_forest",
        })
        assert resp.status_code == 422

    def test_error_response_has_structured_format(self, client):
        """All errors should have error_code + message."""
        resp = client.post("/predict", json={
            "features": {"fake_feature": 1.0},
            "model_type": "random_forest",
        })
        body = resp.json()
        assert "error_code" in body or "detail" in body

    def test_request_id_in_response_header(self, client):
        resp = client.get("/health")
        assert "x-request-id" in resp.headers
