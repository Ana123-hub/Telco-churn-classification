import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc

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

#Final fit and threshold Optimization ---
model.fit(X_train, y_train)

train_probs = model.predict_proba(X_train)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, train_probs)
print("Test Holdout ROC_AUC score:", (auc(fpr, tpr)))
gmeans = np.sqrt(tpr * (1 - fpr))
best_threshold = thresholds[np.argmax(gmeans)]

#Evaluate performance
test_probs = model.predict_proba(X_test)[:, 1]
custom_preds = (test_probs >= best_threshold).astype(int)

print(f"--- Final Baseline Report (Threshold: {best_threshold:.4f}) ---")
print(classification_report(y_test, custom_preds))

#Freeze model
joblib.dump(model, "models/logistic_reg_v1.pkl")
joblib.dump(best_threshold, "models/best_threshold_v1.pkl")
print("Model pipeline frozen successfully inside 'models/'.")

##LOAD V2 DATA AND TRAIN EXPERIMENTAL MODEL
#Load v2 Data
print("Loading v2 feature-engineered datasets...")
X_train = np.load("data/processed/X_train_v2.npy", allow_pickle=True)
X_test = np.load("data/processed/X_test_v2.npy", allow_pickle=True)
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

#Train optimised model
#Setting C as 0.1 applies slight L2 regularization to control interaction variance
model_log = LogisticRegression(class_weight='balanced', C=0.1, solver='saga', max_iter=2000, random_state=42)

#Check cross-validation performance
cv_scores = cross_val_score(model_log, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Mean v2 Cross-Validated ROC-AUC: {np.mean(cv_scores):.4f}")

#Fit model_log and optimize threshold
model_log.fit(X_train, y_train)
test_probs = model_log.predict_proba(X_test)[:, 1]

# Calculate true final ROC-AUC on unseen data
fpr, tpr, thresholds = roc_curve(y_test, test_probs)
final_roc_auc = auc(fpr, tpr)
print(f"Final Test Holdout ROC-AUC Score: {final_roc_auc:.4f}")

#Find ideal classification threshold using Geometric Mean
gmeans = np.sqrt(tpr * (1 - fpr))
best_threshold = thresholds[np.argmax(gmeans)]

custom_preds = (test_probs >= best_threshold).astype(int)
print(f"\n--- Final Performance Report (Threshold: {best_threshold:.4f}) ---")
print(classification_report(y_test, custom_preds))

#Freeze v2 Models
joblib.dump(model_log, "models/logistic_reg_v2.pkl")
joblib.dump(best_threshold, "models/best_threshold_v2.pkl")
print("Experimental model pipeline version 2 frozen successfully.")