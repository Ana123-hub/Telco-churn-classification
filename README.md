# Telco Customer Churn Prediction and Engineering Pipeline

An end-to-end machine learning system built with scikit-learn to predict customer churn. This project showcases the evolution of a production pipeline across two dedicated Git branches—moving from multi-model baseline exploration and domain feature engineering equipped with automated data-defect patching and classification threshold optimization to containerized microservice deployment with Docker.

---

## Architecture and Production Stack

```text
Telco/
├── models/                  # Serialized model artifacts & preprocessors
│   └── logistic_regression_model.pkl
├── src/                     # Core application source code
│   ├── main.py              # FastAPI application server & endpoint handlers
│   ├── predict.py           # Feature transformation & inference pipeline
│   ├── preprocess.py        # Data cleaning, encoding & feature transformers
│   ├── train.py             # Model training & validation script
│   └── config.py            # Environment & model configurations
├── tests/                   # Automated unit & integration test suite
│   ├── test_api.py          # FastAPI endpoint tests
│   ├── test_config.py       # Configuration & path validation tests
│   └── test_preprocess.py   # Preprocessing & transformation unit tests
├── Dockerfile               # Containerization blueprint (Python 3.11-slim)
├── requirements.txt         # Production-only environment dependencies
└── README.md
```
* **API Engine:** FastAPI + Uvicorn
* **Testing Suite:** Pytest
* **Model Runtime:** Python 3.11-slim container environment
* **ML Stack:** Scikit-learn, Pandas, NumPy, Joblib
* **Containerization:** Docker Desktop + WSL 2 backend

---

## Multi-Model Performance Summary

Our models were tracked and evaluated using the Receiver Operating Characteristic Area Under the Curve (ROC-AUC) metric on a stratified test holdout set. 

|  Model     |  Branch     |  Strategy / Focus     |  Test ROC-AUC     |  Target Status     |
| :---       | :---        | :---                  | :---              | :---               |
| **AdaBoost Classifier**  | `main`   | Adaptive boosting ensemble on baseline features   | **0.8408**   |  **Target Achieved**   |
| **Baseline Logistic Regression**   | `main`   | Fast, interpretable linear classification baseline   | **0.8371**   | Near Target   |
| **Experimental Logistic Regression**   | `feature-engineering-v2`   | Regularized linear model with engineered features   | **0.7733**   | Stable / Robust   |
| **Decision Tree Classifier**   | `main`   | Non-linear tree structure baseline   | **0.8276**   | Baseline   |

### Core Insight from Experimentation
While our **Experimental Logistic Regression** added domain-specific features, it hit a linear boundary wall at `0.7733`. The **AdaBoost Classifier** successfully crossed the **0.8408** threshold by sequentially building shallow decision trees that adaptively focus on hard-to-predict customer profiles. For API containerization and initial microservice deployment, the robust **Logistic Regression** baseline was selected for its high speed, small footprint, and high baseline performance (**0.8371**).

---

## Deployment & Containerization (Docker + FastAPI)

The production pipeline is packaged inside a lightweight Docker container to guarantee environment parity across development and production environments.

### 1. Build the Docker Image
```bash
docker build -t telco-churn-api:v1 .
```
### 2. Run the Containerised service.
```bash
docker run -d -p 8000:8000 --name telco-api telco-churn-api:v1
```

---

## API Inference Quickstart (PowerShell)
Send a customer JSON payload to receive real-time prediction probalities:
```text
$body = @{
    gender = "Female"
    SeniorCitizen = 0
    Partner = "Yes"
    Dependents = "No"
    tenure = 1
    PhoneService = "Yes"
    MultipleLines = "No"
    InternetService = "DSL"
    OnlineSecurity = "No"
    OnlineBackup = "Yes"
    DeviceProtection = "No"
    TechSupport = "No"
    StreamingTV = "No"
    StreamingMovies = "No"
    Contract = "Month-to-month"
    PaperlessBilling = "Yes"
    PaymentMethod = "Electronic check"
    MonthlyCharges = 29.85
    TotalCharges = 29.85
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```
## Sample Response:
```text
{
  "status": "success",
  "data": {
    "churn_prediction": 1,
    "churn_label": "Yes",
    "churn_probability": 0.7083
  }
}
```

## Repository Architecture & Workspace

This repository is modularly structured to enforce clean code boundaries, eliminate data leakage, and ensure reproducibility across training and inference states.
