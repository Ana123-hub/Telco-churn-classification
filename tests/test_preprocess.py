import pytest
import pandas as pd
import numpy as np
from src.preprocess import TelcoFeatureEngineer


@pytest.fixture
def sample_data():
    """Creates a mock DataFrame containing all required feature columns for testing."""
    return pd.DataFrame({
        "tenure": [0, 12, 24],
        "MonthlyCharges": [20.0, 70.0, 100.0],
        "TotalCharges": [0.0, 840.0, 2400.0],
        "Contract": ["Month-to-month", "Month-to-month", "Two year"],
        "PhoneService": ["Yes", "Yes", "Yes"],
        "InternetService": ["No", "DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes", "Yes"]
    })


def test_feature_engineer_row_count(sample_data):
    """Ensure feature engineering does not drop or duplicate rows."""
    service_cols = ["PhoneService", "InternetService", "OnlineSecurity"]
    transformer = TelcoFeatureEngineer(service_cols=service_cols)
    
    df_transformed = transformer.fit_transform(sample_data)
    
    assert len(df_transformed) == len(sample_data)


def test_charges_per_tenure_zero_division(sample_data):
    """Verify tenure = 0 does not raise ZeroDivisionError or produce Infinity/NaN."""
    service_cols = ["PhoneService", "InternetService", "OnlineSecurity"]
    transformer = TelcoFeatureEngineer(service_cols=service_cols)
    df_transformed = transformer.fit_transform(sample_data)
    
    # First row has tenure = 0; zero-division safe handling converts tenure=0 to 1
    # 0.0 / 1 = 0.0
    first_row_val = df_transformed.loc[0, "Charges_per_Tenure"]
    
    assert not np.isnan(first_row_val)
    assert not np.isinf(first_row_val)
    assert first_row_val == 0.0


def test_total_services_calculation(sample_data):
    """Verify horizontal counting of active services."""
    service_cols = ["PhoneService", "InternetService", "OnlineSecurity"]
    transformer = TelcoFeatureEngineer(service_cols=service_cols)
    df_transformed = transformer.fit_transform(sample_data)
    
    # Row 0: PhoneService="Yes", InternetService="No", OnlineSecurity="No" -> Count = 1
    # Row 1: "Yes", "DSL", "Yes" -> Count = 3
    assert df_transformed.loc[0, "Total_Services"] == 1
    assert df_transformed.loc[1, "Total_Services"] == 3