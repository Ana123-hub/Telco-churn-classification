import pytest
from fastapi.testclient import TestClient
from src.main import app

# Initialize the TestClient with our FastAPI app
client = TestClient(app)


def test_health_check():
    """Verify that the GET / health check endpoint returns 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "Telco Churn API"}


def test_predict_endpoint_success():
    """Verify that a valid customer payload receives a 200 OK and prediction response."""
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    
    json_data = response.json()
    assert json_data["status"] == "success"
    
    data = json_data["data"]
    assert "churn_prediction" in data
    assert "churn_label" in data
    assert "churn_probability" in data
    
    # Value assertions
    assert data["churn_prediction"] in [0, 1]
    assert data["churn_label"] in ["Yes", "No"]
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_endpoint_missing_field():
    """Verify that missing required fields trigger a 422 Unprocessable Entity validation error."""
    incomplete_payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        # 'tenure' and other fields intentionally omitted
    }

    response = client.post("/predict", json=incomplete_payload)
    
    assert response.status_code == 422