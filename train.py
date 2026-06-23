import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_curve

#Load the processed data 
print("Loading preprocessed numpy datasets...")
X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

#Initialize and cross-validate baseline
print("Training Baseline Logistic Regression Model...")
model = LogisticRegression(class_weight='balanced', random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Mean Baseline 5-Fold ROC-AUC: {np.mean(cv_scores):.4f}\n")

# --- 3. Final Fit and Threshold Optimization ---
model.fit(X_train, y_train)

train_probs = model.predict_proba(X_train)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, train_probs)
gmeans = np.sqrt(tpr * (1 - fpr))
best_threshold = thresholds[np.argmax(gmeans)]

# --- 4. Evaluate Performance ---
test_probs = model.predict_proba(X_test)[:, 1]
custom_preds = (test_probs >= best_threshold).astype(int)

print(f"--- Final Baseline Report (Threshold: {best_threshold:.4f}) ---")
print(classification_report(y_test, custom_preds))

# --- 5. Freeze Model Artifacts ---
joblib.dump(model, "models/logistic_reg_v1.pkl")
joblib.dump(best_threshold, "models/best_threshold_v1.pkl")
print("Model pipeline frozen successfully inside 'models/'.")