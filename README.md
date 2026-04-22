# Wind Turbine Health Classifier (WTHC)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF9900.svg)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete machine learning pipeline and production-ready FastAPI serving layer for predicting wind turbine anomalies based on SCADA sensor data.

> **Disclaimer:** This repository is specifically designed as the production inference layer for wind turbine structural health. **It is NOT a pre-trained universal model.** Before deploying, you must train the models (Logistic Regression & Random Forest) on your own personal SCADA `.csv` datasets using the provided pipeline scripts to generate the `.joblib` model artifacts. Once trained on your specific turbine sensor mappings, the API can be plugged in to serve real-time predictions.

This project trains and serves three models on ~200 numerical sensor features (including temperatures, wind speed, vibrations, voltages):
- **Logistic Regression**: Linear baseline with L1 Regularization (feature selection).
- **Random Forest**: High-capacity tree-based classifier for complex, non-linear sensor patterns.
- **XGBoost**: State-of-the-art gradient boosted trees for maximizing the precision-recall boundary.

### 📈 Uncapped Scalability (>250 Sensors)
The architecture is inherently built to absorb massive data streams. If you integrate additional hardware (e.g., 250+ advanced turbine sensors), the tree-based models (XGBoost/RF) will non-linearly isolate the strongest predictors, while the API will automatically parse `.joblib` metadata to dynamically expand the expected JSON payload without requiring any source code modifications.

--- 

## 📂 Project Structure

```text
Wind-turbine-health-classifier/
├── src/
│   ├── api/                 # FastAPI web server and routes
│   │   ├── main.py          # App initialization and logging
│   │   ├── routes.py        # API endpoints
│   │   └── schemas.py       # Pydantic validation models
│   ├── models/              # ML Pipeline logic
│   │   ├── model.py         # Sklearn wrappers for LR and RF
│   │   └── preprocessing.py # Scaling, feature engineering
│   └── config.py            # Hyperparameter configuration (env variables)
├── models/                  # Serialized Scikit-Learn .joblib artifacts go here
├── tests/                   # Extensive test suite
├── docker-compose.yml       # Docker configuration
├── Dockerfile               # Container build instructions
└── requirements.txt         # Python dependencies
```

---

## 🚀 Setup & Installation

### Option A: Running via Docker (Recommended)

Docker is the easiest way to run the API without installing Python dependencies on your local machine.

1. Make sure you have [Docker](https://docs.docker.com/get-docker/) installed.
2. Ensure you have the model artifacts (`.joblib` files) in the `models/` directory (see [Model Management](#-model-management) below).
3. From the root of the project, run:
   ```bash
   docker-compose up --build
   ```
4. The API will be available at `http://localhost:8000`.

### Option B: Local Development Setup

If you prefer to run the application locally or want to contribute to the code:

1. **Install Python 3.11+**
2. **Setup Virtual Environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate it (Linux/Mac)
   source venv/bin/activate  
   
   # Activate it (Windows)
   venv\Scripts\activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the FastAPI App:**
   ```bash
   # Run the server with hot-reloading
   uvicorn src.api.main:app --reload
   ```

---

## 🧪 How to Run the Tests

The project uses `pytest` for executing unit tests, integration tests against the FastAPI client, and performance regression tests against the saved models.

With your virtual environment active, run:
```bash
python -m pytest tests/ -v
```

---

## 📡 API Usage & Endpoints

Once the application is running, you can view the fully interactive Swagger API documentation in your browser at: 
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

Below are examples of how to use the specific API endpoints.

### 1. Check API Health (`GET /health`)
Validates that the FastAPI application is responding and lists successfully loaded `.joblib` model artifacts.

**Example Request:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Example Response:**
```json
{
  "status": "healthy",
  "loaded_models": ["logistic_regression", "random_forest"],
  "models_dir": "/app/models"
}
```

### 2. View Model Info (`GET /model-info`)
Retrieve the full model card for a given model type. Returns hyper parameters, training metrics, and the exact expected feature schema.

**Example Request:**
```bash
curl -X GET "http://localhost:8000/model-info?model_type=random_forest"
```

### 3. Make a Single Prediction (`POST /predict`)
Submit a single JSON observation containing your sensor readings.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
          "model_type": "xgboost",
          "features": {
            "Wind speed 1 (avg)": 5.2,
            "Rotor speed 1 (avg)": 12.3,
            "Nacelle outside temperature (avg)": 8.1
            // ... include all required features here
          }
        }'
```

**Example Response:**
```json
{
  "prediction": "Normal",
  "anomaly_probability": 0.0452,
  "model_type": "xgboost",
  "model_version": 2
}
```

### 4. Make Batch Predictions (`POST /batch-predict`)
Submit a list of observations to be scored in a single vectorized pass. Max 1000 samples per request.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/batch-predict" \
     -H "Content-Type: application/json" \
     -d '{
          "model_type": "xgboost",
          "samples": [
             { "Wind speed 1 (avg)": 5.2, "Rotor speed 1 (avg)": 12.3 },
             { "Wind speed 1 (avg)": 18.5, "Rotor speed 1 (avg)": 28.1 }
          ]
        }'
```

---

## 🤖 Model Management

Serialized Scikit-Learn model artifacts (`.joblib` files) are stored in the `/models` directory.

The FastAPI application scans this directory at startup and automatically loads the highest version number of each model type found (e.g. `random_forest_v1.joblib`).

> **CRITICAL**: The GitHub repository uses `.gitkeep` as a placeholder for the `models/` directory because model files are often too large for version control. **Before you can use the `/predict` endpoints, you must execute the model training pipeline to generate the `.joblib` files and place them in the `models/` folder.**

## ⚙️ Configuration Variables

The application behavior can be customized using environment variables. To change these locally, you can modify the `docker-compose.yml` environment section, or export them in your shell before running locally.

**Machine Learning Overrides (Default fallbacks shown below):**

### XGBoost Defaults (Optuna Optimized):
* `XGB_LEARNING_RATE` = 0.02146
* `XGB_MAX_DEPTH` = 7
* `XGB_N_ESTIMATORS` = 400
* `XGB_SCALE_POS_WEIGHT` = 7.25

### Logistic Regression & Random Forest Defaults:
* `LR_C` = 0.1
* `LR_MAX_ITER` = 100
* `RF_N_ESTIMATORS` = 100
* `RF_MAX_DEPTH` = 20

---

## 🛠️ CI/CD Pipeline
This project includes a GitHub Actions configuration located in `.github/workflows/deploy.yml`. 

On every PR or Push to the `main` branch, GitHub Actions will:
1. Setup a Python 3.11 environment.
2. Install dependencies.
3. Run the complete test suite.
4. Verify if the Docker image can be successfully built.
