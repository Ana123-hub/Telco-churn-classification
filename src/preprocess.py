import os
import joblib
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
try:
    from src.config_loader import load_config
except ModuleNotFoundError:
    from config_loader import load_config

config = load_config()
raw_data_path = config["abs_paths"]["raw_data"]
c_param = config["model_params"]["C"]
target_col = config["data_cleaning"]["target_column"]

#Get data path directly from inside active terminal workspace root folder
ROOT_DIR = r"C:\Users\MY-PC\Documents\Telco"

DATA_PATH = os.path.join(ROOT_DIR, "Data", "raw", "telco_customer_churn.csv")

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)


#seperate features and target variable, drop irrelevant columns
X = df.drop(columns=["Churn", "customerID", "TotalCharges", "Partner", "PhoneService", "gender"])
y = df["Churn"].map({"No": 0, "Yes": 1}) #ensure target is binary integers

#Train/test split (Stratified for class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Define feature columns
categorical_cols = [
    "Contract", "PaperlessBilling", "InternetService", "OnlineSecurity", 
    "PaymentMethod", "TechSupport", "Dependents", "OnlineBackup", 
    "MultipleLines", "DeviceProtection", "StreamingTV", "StreamingMovies"
]

numerical_cols = ["MonthlyCharges", "tenure", "SeniorCitizen"]

#Initialise the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", TargetEncoder(smooth="auto", cv=5, random_state=42), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

#Fit on Train data Only and transform both on Test
print("Fitting preprocessing pipelines...")
X_train_processed = preprocessor.fit_transform(X_train, y_train)
X_test_processed = preprocessor.transform(X_test)

#Freeze and save preprocessor and split data
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

#Save the transformer object so we can use it during inference
joblib.dump(preprocessor, "models/preprocessor_v1.pkl")

# Save the processed numpy arrays for the training script to fetch
np.save("data/processed/X_train.npy", X_train_processed)
np.save("data/processed/X_test.npy", X_test_processed)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Preprocessing complete. Artifacts 'frozen' and saved successfully.")

##FEATURE ENGINEERING
#Custom transformer for engineering Charges_per_tenure, 
#High_cost_month_to_month and Total_services new features
class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, service_cols=None):
        self.service_cols = service_cols
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_out = X.copy()
        
        # Total_Services count (Safely handle missing columns during testing or API calls)
        if self.service_cols:
            # Filter to keep only columns that exist in X_out
            existing_services = [col for col in self.service_cols if col in X_out.columns]
            if existing_services:
                X_out["Total_Services"] = (X_out[existing_services] != "No").sum(axis=1)
            else:
                X_out["Total_Services"] = 0
        else:
            X_out["Total_Services"] = 0

        #Charges_per_Tenure (with zero-division protection)
        tenure_safe = np.where(X_out["tenure"] == 0, 1, X_out["tenure"])
        X_out["Charges_per_Tenure"] = X_out["TotalCharges"] / tenure_safe

        # High_Cost_Month_to_Month binary indicator
        if "Contract" in X_out.columns and "MonthlyCharges" in X_out.columns:
            median_monthly = X_out["MonthlyCharges"].median()
            X_out["High_Cost_Month_to_Month"] = (
                (X_out["Contract"] == "Month-to-month") & 
                (X_out["MonthlyCharges"] > median_monthly)
            ).astype(int)
        else:
            X_out["High_Cost_Month_to_Month"] = 0

        return X_out

#Load raw dataset
ROOT_DIR = r"C:\Users\MY-PC\Documents\Telco"
DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "telco_customer_churn.csv")
df = pd.read_csv(DATA_PATH)

#clean empty spaces in TotalCharges column
df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
df['TotalCharges'] = df['TotalCharges'].fillna(0).astype(float)

#Drop irrelevant columns and target variable, exclude TotalCharges as we will engineer a new feature from it
X = df.drop(columns=["Churn", "customerID", "Partner", "PhoneService", "gender"])
y = df["Churn"].map({"No": 0, "Yes": 1})

#Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Apply feature engineering
engineer = TelcoFeatureEngineer()
X_train_eng = engineer.transform(X_train)
X_test_eng = engineer.transform(X_test)

#Define column pools, including newly engineered features
categorical_cols = [
    "Contract", "PaperlessBilling", "InternetService", "OnlineSecurity", 
    "PaymentMethod", "TechSupport", "Dependents", "OnlineBackup", 
    "MultipleLines", "DeviceProtection", "StreamingTV", "StreamingMovies"
]

#Add newly created continuous/numerical variables
#Exclude high_cost_month_to_month
numerical_cols = ["MonthlyCharges", "tenure", "SeniorCitizen", 
                  "Charges_per_Tenure", "Total_Services"]

#Note: High_Cost_Month_To_Month is a binary flag (0 or 1) already, 
#so we pass it through without scaling or encoding
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", TargetEncoder(smooth="auto", cv=5, random_state=42), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ],
    remainder="passthrough" 
)

#Fit and transform
print("Executing feature engineering pipeline...")
X_train_cleaned = preprocessor.fit_transform(X_train_eng, y_train).astype(np.float64)
X_test_cleaned = preprocessor.transform(X_test_eng).astype(np.float64)

#Freeze and save
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

joblib.dump(preprocessor, "models/preprocessor_v2.pkl")
np.save("data/processed/X_train_v2.npy", X_train_cleaned)
np.save("data/processed/X_test_v2.npy", X_test_cleaned)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Step 1 Complete: Feature engineering arrays saved as v2.")

