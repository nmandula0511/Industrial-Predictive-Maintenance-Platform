import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import Config
from src.pipeline.feature_engineering import FeatureEngineer
from src.pipeline.data_loader import load_dataset, load_rul_labels

class InferenceEngine:
    def __init__(self):
        self.model_path = Config.MODEL_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model not found at {self.model_path}. Please run train.py first.")
            
        self.model = joblib.load(self.model_path)
        self.fe = FeatureEngineer()
        self.fe.load_artifacts(Config.MODELS_DIR)
        
    def predict(self, raw_df: pd.DataFrame) -> np.ndarray:
        """Processes raw telemetry data (e.g. time series sequence) and returns RUL predictions for each row."""
        # Standardize columns
        if len(raw_df.columns) < len(Config.ALL_COLUMNS):
            # If input lacks columns, let's make sure it contains engine_id, cycle, and sensors
            # If it's a sub-dataframe with specific sensors, raise error or pad. We assume standard structure.
            pass
            
        # Transform (rolling + scaling)
        X_scaled = self.fe.transform(raw_df)
        preds = self.model.predict(X_scaled)
        
        # RUL cannot be negative in practice
        preds = np.clip(preds, 0, None)
        return preds

    def evaluate_test_set(self) -> dict:
        """Evaluates model on the test dataset using the CMAPSS evaluation protocol.
        
        We predict the RUL at the LAST recorded cycle for each engine and compare it 
        with the ground truth RUL in RUL_FD001.txt.
        """
        print("[*] Loading test data...")
        test_df = load_dataset(Config.TEST_DATA_PATH)
        rul_labels = load_rul_labels(Config.RUL_DATA_PATH)
        
        # Sort and run feature engineering on the whole test set to compute rolling stats correctly
        test_df = test_df.sort_values(by=["engine_id", "cycle"])
        X_test_all = self.fe.transform(test_df)
        
        # Predict on all points
        preds_all = self.model.predict(X_test_all)
        preds_all = np.clip(preds_all, 0, None)
        
        # Add predictions and engine_id to locate final cycles
        test_df["predicted_RUL"] = preds_all
        
        # Get the last record for each engine
        last_records = test_df.groupby("engine_id").last().reset_index()
        
        # Get actual and predicted values
        y_true = rul_labels
        # Make sure counts match
        n_engines = len(last_records)
        if len(y_true) != n_engines:
            # Slice to match
            y_true = y_true[:n_engines]
            
        y_pred = last_records["predicted_RUL"].values
        
        # Metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Custom CMAPSS Scoring Function: asymmetric penalty (overestimating is penalized more than underestimating)
        # s = sum(e^(d/13)-1) if d < 0, else sum(e^(d/10)-1) where d = predicted - true
        diff = y_pred - y_true
        scores = []
        for d in diff:
            if d < 0:
                scores.append(np.exp(-d / 13.0) - 1.0)
            else:
                scores.append(np.exp(d / 10.0) - 1.0)
        cmapss_score = np.sum(scores)
        
        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "cmapss_score": cmapss_score,
            "predictions": y_pred.tolist(),
            "actuals": y_true.tolist()
        }

if __name__ == "__main__":
    print("Testing Inference Engine...")
    try:
        engine = InferenceEngine()
        metrics = engine.evaluate_test_set()
        print("\nTEST SET EVALUATION RESULTS:")
        print(f"  RMSE:        {metrics['rmse']:.4f}")
        print(f"  MAE:         {metrics['mae']:.4f}")
        print(f"  R2 Score:    {metrics['r2']:.4f}")
        print(f"  CMAPSS Score:{metrics['cmapss_score']:.4f}")
    except Exception as e:
        print("Error during inference test:", e)
        print("Ensure the model has been trained by running src/pipeline/train.py first.")
