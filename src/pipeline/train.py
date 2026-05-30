import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import Config
from src.pipeline.data_loader import load_dataset
from src.pipeline.feature_engineering import FeatureEngineer

import argparse
import time

def run_training(dataset_type: str = "cmapss"):
    print("=" * 60)
    print("STARTING MODEL TRAINING PIPELINE")
    print(f"Target Dataset: {dataset_type.upper()}")
    print("=" * 60)
    
    overall_start = time.time()
    
    # 1. Determine dataset file path
    if dataset_type == "large":
        data_path = os.path.join(Config.DATA_DIR, "fleet_telemetry_large.csv")
        if not os.path.exists(data_path):
            print("[!] Large fleet telemetry data file not found on disk.")
            print("[*] Initiating auto-generation of 500 engines telemetry...")
            from src.pipeline.generator import generate_fleet_telemetry
            generate_fleet_telemetry(num_engines=500, output_path=data_path)
    else:
        data_path = Config.TRAIN_DATA_PATH
        
    # Load dataset
    print(f"[*] Loading training data from: {data_path}")
    t0 = time.time()
    train_df = load_dataset(data_path)
    load_time = time.time() - t0
    print(f"[+] Data loaded. Shape: {train_df.shape} | Time: {load_time:.2f}s")
    
    # 2. Preprocess & Feature Engineer
    print("[*] Performing feature engineering & computing target RUL...")
    t0 = time.time()
    fe = FeatureEngineer(window_size=10)
    X, y = fe.fit_transform(train_df)
    fe_time = time.time() - t0
    print(f"[+] Features constructed. Shape: {X.shape} | Time: {fe_time:.2f}s")
    
    # 3. Train/Validation Split for local performance check
    print("[*] Creating train/validation splits...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    print(f"    Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    
    # 4. Train Model
    print(f"[*] Training RandomForestRegressor with {Config.N_ESTIMATORS} estimators (n_jobs=-1)...")
    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=Config.N_ESTIMATORS,
        max_depth=15,
        min_samples_leaf=5,
        random_state=Config.RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    print(f"[+] Model training completed. | Time: {fit_time:.2f}s")
    
    # 5. Evaluate Validation Set
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    mae = mean_absolute_error(y_val, val_preds)
    r2 = r2_score(y_val, val_preds)
    
    print("-" * 40)
    print("VALIDATION METRICS:")
    print(f"  RMSE: {rmse:.4f} cycles")
    print(f"  MAE:  {mae:.4f} cycles")
    print(f"  R2:   {r2:.4f}")
    print("-" * 40)
    
    # 6. Train on Full Data and Save Artifacts
    print("[*] Fitting final model on full training set...")
    t0 = time.time()
    final_model = RandomForestRegressor(
        n_estimators=Config.N_ESTIMATORS,
        max_depth=15,
        min_samples_leaf=5,
        random_state=Config.RANDOM_STATE,
        n_jobs=-1
    )
    final_model.fit(X, y)
    
    print(f"[*] Saving model artifact to: {Config.MODEL_PATH}")
    joblib.dump(final_model, Config.MODEL_PATH)
    
    print(f"[*] Saving feature engineering scaler and schema to: {Config.MODELS_DIR}")
    fe.save_artifacts(Config.MODELS_DIR)
    save_time = time.time() - t0
    
    # 7. Print Feature Importances
    importances = final_model.feature_importances_
    feat_importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    print("\nTOP 10 CRITICAL SENSORS / FEATURES:")
    print(feat_importance_df.head(10).to_string(index=False))
    
    overall_time = time.time() - overall_start
    print("=" * 60)
    print("TRAINING PIPELINE COMPLETE SUCCESSFULLY")
    print(f"  Total pipeline execution time: {overall_time:.2f} seconds")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industrial ML Pipeline Trainer")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["cmapss", "large"], 
        default="cmapss", 
        help="Dataset size to train on: 'cmapss' (standard NASA) or 'large' (simulated fleet)"
    )
    args = parser.parse_args()
    run_training(args.dataset)
