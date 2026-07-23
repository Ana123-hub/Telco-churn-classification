import os
import joblib
import pandas as pd
import numpy as np
try:
    from src.config_loader import load_config
except ModuleNotFoundError:
    from config_loader import load_config


class ChurnPredictor:
    """Inference class using preprocessor_v1 and baseline logistic_reg_v1."""

    def __init__(self):
        config = load_config()
        models_dir = config["abs_paths"]["models_dir"]

        pipeline_path = os.path.join(models_dir, "preprocessor_v1.pkl")
        model_path = os.path.join(models_dir, "logistic_reg_v1.pkl")

        if not os.path.exists(pipeline_path) or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Artifacts missing. Please ensure '{pipeline_path}' and '{model_path}' exist."
            )

        self.preprocessor = joblib.load(pipeline_path)
        self.model = joblib.load(model_path)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds engineered features expected by preprocessor_v1."""
        df_feats = df.copy()

        # 1. Charges_per_Tenure
        df_feats["Charges_per_Tenure"] = np.where(
            df_feats["tenure"] > 0,
            df_feats["TotalCharges"] / df_feats["tenure"],
            0.0
        )

        # 2. Total_Services
        service_cols = [
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        active_counts = pd.Series(0, index=df_feats.index)
        for col in service_cols:
            if col in df_feats.columns:
                active_counts += df_feats[col].astype(str).isin(["Yes", "DSL", "Fiber optic"]).astype(int)
        df_feats["Total_Services"] = active_counts

        # 3. High_Cost_Month_to_Month
        is_m2m = df_feats["Contract"] == "Month-to-month"
        is_high_cost = df_feats["MonthlyCharges"] > 70.0
        df_feats["High_Cost_Month_to_Month"] = (is_m2m & is_high_cost).astype(int)

        return df_feats

    def predict(self, input_df: pd.DataFrame):
        """Engineers features, transforms input, and generates predictions."""
        # 1. Add required engineered columns
        engineered_df = self._engineer_features(input_df)

        # 2. Transform through preprocessor_v1
        X_processed = self.preprocessor.transform(engineered_df)

        # 3. Predict with baseline logistic_reg_v1
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)[:, 1]

        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                "churn_prediction": int(pred),
                "churn_label": "Yes" if pred == 1 else "No",
                "churn_probability": round(float(prob), 4),
            })

        return results