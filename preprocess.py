import os
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer

#Get data path directly from inside active terminal workspace root folder
ROOT_DIR = r"C:\Users\MY-PC\Documents\Telco"

DATA_PATH = os.path.join(ROOT_DIR, "Data", "raw", "telco_customer_churn.csv")

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)


#seperate features and target variable, drop irrelevant columns
X = df.drop(columns=["Churn", "customerID", "TotalCharges", "Partner", "PhoneService", "gender"])
y = df["Churn"].map({"No": 0, "Yes": 1}) #ensure target is binary integers

#Train/Test Split (Stratified for Class Imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. Define Feature Columns ---
categorical_cols = [
    "Contract", "PaperlessBilling", "InternetService", "OnlineSecurity", 
    "PaymentMethod", "TechSupport", "Dependents", "OnlineBackup", 
    "MultipleLines", "DeviceProtection", "StreamingTV", "StreamingMovies"
]

numerical_cols = ["MonthlyCharges", "tenure", "SeniorCitizen"]

#Construct the Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", TargetEncoder(), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

#Fit on Train data Only and transform both on Test
print("Fitting preprocessing pipelines...")
X_train_processed = preprocessor.fit_transform(X_train, y_train)
X_test_processed = preprocessor.transform(X_test)

#Freeze and Save Preprocessor and Split Data ---
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Save the transformer object so we can use it during inference
joblib.dump(preprocessor, "models/preprocessor_v1.pkl")

# Save the processed numpy arrays for the training script to fetch
np.save("data/processed/X_train.npy", X_train_processed)
np.save("data/processed/X_test.npy", X_test_processed)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Preprocessing complete. Artifacts 'frozen' and saved successfully.")

