import os
import joblib
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

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
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_out = X.copy()
        
        #To avoid division by zero for new customers with 0 tenure, replace with 1
        tenure_safe = X_out['tenure'].replace(0, 1)
        X_out['Charges_per_Tenure'] = X_out['MonthlyCharges'] / tenure_safe
        
        #Select highly volatile month-to-month contracts with high charges
        X_out['High_Cost_Month_to_Month'] = ((X_out['Contract'] == 'Month-to-month') & 
                                             (X_out['MonthlyCharges'] > 70)).astype(int)
        
        #Count total internet services activated for each user by row
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies']
        X_out['Total_Services'] = X_out[service_cols].apply(
            lambda row: sum(row == 'Yes'), axis=1
        )
        
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

