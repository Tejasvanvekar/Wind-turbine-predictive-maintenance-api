"""
Wind Turbine Health Classifier — FastAPI Application
=====================================================
Application factory, startup model loading, structured logging,
request middleware, global exception handlers, and entry point.
"""

import os
import re
import glob
import time
import uuid
import logging
import sys
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .routes import router


# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

def setup_logging():
    """
    Configure structured logging for the application.

    Log format: [TIMESTAMP] LEVEL — LOGGER — MESSAGE
    All application logs use the 'wthc' namespace.
    """
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # Root logger for our app namespace
    app_logger = logging.getLogger("wthc")
    app_logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on reload
    if not app_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        app_logger.addHandler(handler)

    # Quiet down noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    return app_logger


logger = setup_logging()


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Models directory — resolved relative to this file's location.
# In production layout: Wind-turbine-health-classifier/models/
_API_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_API_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

MODEL_FILE_EXTENSION = ".joblib"

# In-memory store for loaded model artifacts
# Key: model_type string (e.g. "random_forest")
# Value: full artifact dict {"model": ..., "scaler": ..., "metadata": {...}}
loaded_models: dict = {}


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def _find_latest_models() -> dict[str, str]:
    """
    Scan the models directory and find the latest version of each model type.

    Returns a dict of {model_type: filepath} for the highest versioned file
    of each model type found.
    """
    pattern = os.path.join(MODELS_DIR, f"*{MODEL_FILE_EXTENSION}")
    files = glob.glob(pattern)

    # Group by model type, track highest version
    latest: dict[str, tuple[int, str]] = {}

    for filepath in files:
        basename = os.path.basename(filepath)
        match = re.match(r"(.+)_v(\d+)\.joblib$", basename)
        if not match:
            continue

        model_type = match.group(1)
        version = int(match.group(2))

        if model_type not in latest or version > latest[model_type][0]:
            latest[model_type] = (version, filepath)

    return {mt: path for mt, (_, path) in latest.items()}


def load_models_from_disk():
    """Load the latest version of each model type into memory."""
    global loaded_models
    loaded_models.clear()

    if not os.path.isdir(MODELS_DIR):
        logger.warning("Models directory not found: %s", MODELS_DIR)
        return

    latest = _find_latest_models()

    if not latest:
        logger.warning("No model files found in: %s", MODELS_DIR)
        return

    for model_type, filepath in latest.items():
        try:
            artifact = joblib.load(filepath)
            loaded_models[model_type] = artifact
            meta = artifact.get("metadata", {})
            logger.info(
                "Loaded %s v%s (%s) from %s",
                model_type, meta.get("model_version", "?"),
                meta.get("model_class", "unknown"), filepath
            )
        except Exception as e:
            logger.error("Failed to load %s: %s", filepath, e)

    logger.info("%d model(s) loaded successfully.", len(loaded_models))


# ==============================================================================
# APPLICATION LIFESPAN
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    logger.info("=" * 60)
    logger.info("STARTING WIND TURBINE HEALTH CLASSIFIER API")
    logger.info("=" * 60)
    logger.info("Models directory: %s", MODELS_DIR)
    load_models_from_disk()
    yield
    # Shutdown
    loaded_models.clear()
    logger.info("API shutdown complete.")


# ==============================================================================
# FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="Wind Turbine Health Classifier",
    description=(
        "REST API for wind turbine anomaly detection. "
        "Serves trained ML models (Logistic Regression, Random Forest) "
        "that classify turbine health as Normal or Anomalous based on "
        "~200 SCADA sensor features."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ==============================================================================
# REQUEST LOGGING MIDDLEWARE
# ==============================================================================

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    Log every request with:
    - Unique request ID (for tracing)
    - Method, path, status code
    - Response time in milliseconds
    """
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()

    logger.info(
        "[%s] --> %s %s",
        request_id, request.method, request.url.path
    )

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "[%s] <-- %s %s %d (%.1fms)",
        request_id, request.method, request.url.path,
        response.status_code, elapsed_ms
    )

    # Attach request ID to response headers for client-side tracing
    response.headers["X-Request-ID"] = request_id
    return response


# ==============================================================================
# GLOBAL EXCEPTION HANDLERS
# ==============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle all HTTPException errors (404, 422, etc.) from our routes.
    If the detail is a structured dict (with error_code), unwrap it to the
    top level for a consistent response format across all error types.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # If our routes raised a structured error dict, use it directly
    if isinstance(exc.detail, dict) and "error_code" in exc.detail:
        content = exc.detail
    else:
        # Wrap plain-string details in our standard format
        content = {
            "error_code": "HTTP_ERROR",
            "message": str(exc.detail),
            "details": None,
        }

    logger.warning(
        "[%s] HTTP %d on %s %s: %s",
        request_id, exc.status_code, request.method,
        request.url.path, content.get("error_code")
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=content,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic / FastAPI request validation errors.
    Returns a structured error response instead of the default 422 format.
    """
    errors = exc.errors()
    logger.warning(
        "Request validation failed on %s %s: %d error(s)",
        request.method, request.url.path, len(errors)
    )

    # Build a user-friendly summary of each error
    formatted_errors = []
    for err in errors:
        field = " → ".join(str(loc) for loc in err.get("loc", []))
        formatted_errors.append({
            "field": field,
            "message": err.get("msg", "Unknown error"),
            "type": err.get("type", "unknown"),
        })

    return JSONResponse(
        status_code=422,
        content={
            "error_code": "VALIDATION_ERROR",
            "message": "Request validation failed. Check the 'details' field for specifics.",
            "details": formatted_errors,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler for unhandled exceptions.
    Logs the full traceback and returns a generic 500 error.
    Never leaks internal details to the client.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(
        "[%s] Unhandled exception on %s %s",
        request_id, request.method, request.url.path
    )

    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
            "details": {"request_id": request_id},
        },
    )


# ==============================================================================
# REGISTER ROUTER
# ==============================================================================

app.include_router(router)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "Wind-turbine-health-classifier.src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
