import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import Config

class FeatureEngineer:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.feature_cols = []
        
    def add_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Remaining Useful Life (RUL) for training dataset."""
        df = df.copy()
        # Find maximum cycle for each engine
        max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
        max_cycles.columns = ["engine_id", "max_cycle"]
        # Merge back
        df = df.merge(max_cycles, on="engine_id")
        # RUL is max_cycle - current_cycle
        df["RUL"] = df["max_cycle"] - df["cycle"]
        df = df.drop(columns=["max_cycle"])
        return df

    def downcast_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcasts numeric types to float32 and int32 to optimize memory footprint by 50%."""
        for col in df.columns:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)
            elif df[col].dtype == np.int64:
                df[col] = df[col].astype(np.int32)
        return df

    def compute_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes rolling mean and standard deviation in a single vectorized pass per engine."""
        df = df.copy()
        # Sort values to ensure correct chronological order
        df = df.sort_values(by=["engine_id", "cycle"])
        
        # Optimize grouping: execute rolling operations on all selected sensors at once
        features_to_roll = Config.SELECTED_SENSORS
        rolled = (
            df.groupby("engine_id")[features_to_roll]
            .rolling(window=self.window_size, min_periods=1)
        )
        
        # Compute stats in parallel/vectorized passes
        rolled_mean = rolled.mean().reset_index(level=0, drop=True)
        rolled_std = rolled.std().reset_index(level=0, drop=True).fillna(0.0)
        
        # Rename columns to match schema
        rolled_mean.columns = [f"{col}_roll_mean" for col in rolled_mean.columns]
        rolled_std.columns = [f"{col}_roll_std" for col in rolled_std.columns]
        
        # Concatenate features back to original df
        df = pd.concat([df, rolled_mean, rolled_std], axis=1)
        
        # Downcast columns to save 50% RAM
        df = self.downcast_df(df)
        return df

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Calculates features, targets, fits the scaler, and returns scaled training features and target."""
        # 1. Add RUL target
        df_with_rul = self.add_rul(df)
        
        # 2. Add rolling features
        df_engineered = self.compute_rolling_features(df_with_rul)
        
        # Determine feature columns (all sensor readings + their rolling stats)
        all_feature_candidates = [col for col in df_engineered.columns if "sensor_" in col]
        # Restrict to selected sensors and their rolling stats
        self.feature_cols = [col for col in all_feature_candidates if any(s in col for s in Config.SELECTED_SENSORS)]
        
        X = df_engineered[self.feature_cols]
        y = df_engineered["RUL"]
        
        # Fit scaler
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_cols,
            index=X.index
        ).astype(np.float32)
        
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms validation or test dataset using rolling features and pre-fitted scaler."""
        df_engineered = self.compute_rolling_features(df)
        
        # Ensure we have all columns expected
        X = df_engineered[self.feature_cols]
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_cols,
            index=X.index
        ).astype(np.float32)
        return X_scaled

    def save_artifacts(self, model_dir: str):
        """Saves scaler and feature list for deployment."""
        os.makedirs(model_dir, exist_ok=True)
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        features_path = os.path.join(model_dir, "feature_cols.json")
        
        joblib.dump(self.scaler, scaler_path)
        
        import json
        with open(features_path, "w") as f:
            json.dump(self.feature_cols, f)
            
    def load_artifacts(self, model_dir: str):
        """Loads scaler and feature list for prediction."""
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        features_path = os.path.join(model_dir, "feature_cols.json")
        
        if not os.path.exists(scaler_path) or not os.path.exists(features_path):
            raise FileNotFoundError("Feature engineering artifacts not found. Please train the model first.")
            
        self.scaler = joblib.load(scaler_path)
        
        import json
        with open(features_path, "r") as f:
            self.feature_cols = json.load(f)
