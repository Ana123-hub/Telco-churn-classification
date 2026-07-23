from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from typing import List
from src.predict import ChurnPredictor

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Production ML API for real-time customer churn inference.",
    version="1.0.0"
)

#Initialize predictor instance on startup
try:
    predictor = ChurnPredictor()
except Exception as e:
    predictor = None
    print(f"Warning: Predictor failed to initialize. {e}")


class CustomerPayload(BaseModel):
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="DSL")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=29.85)
    TotalCharges: float = Field(..., example=358.20)

@app.get("/")
def read_root():
    return {"message": "Telco Churn API is running!"}

@app.get("/health")
def health_check():
    """Simple API health check endpoint."""
    return {"status": "healthy", "service": "Telco Churn API"}


@app.post("/predict")
def predict_churn(customer: CustomerPayload):
    """
    Accepts customer profile data and returns real-time churn prediction.
    """
    try:
        # Convert Pydantic object to Pandas DataFrame
        input_data = pd.DataFrame([customer.model_dump()])
        
        # Perform prediction
        predictions = predictor.predict(input_data)
        
        return {
            "status": "success",
            "data": predictions[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))