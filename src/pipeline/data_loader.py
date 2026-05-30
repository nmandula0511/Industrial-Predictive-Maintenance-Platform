import pandas as pd
import numpy as np
import os
import sys

# Add root folder to sys.path so config can be imported easily
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import Config

def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads a CMAPSS text data file or simulated CSV, standardizing column mapping."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
    if file_path.lower().endswith(".csv"):
        # Load structured CSV
        df = pd.read_csv(file_path)
        # Ensure all columns in ALL_COLUMNS are present
        missing_cols = [c for c in Config.ALL_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV file is missing expected columns: {missing_cols}")
        return df[Config.ALL_COLUMNS]
        
    # Read space-delimited text file
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    
    # Drop columns that are completely null
    df = df.dropna(axis=1, how="all")
    
    # Slice or pad columns to match the expected number of config columns
    num_cols = len(Config.ALL_COLUMNS)
    if df.shape[1] > num_cols:
        df = df.iloc[:, :num_cols]
    elif df.shape[1] < num_cols:
        raise ValueError(f"Dataset at {file_path} only has {df.shape[1]} columns, expected {num_cols}.")
        
    df.columns = Config.ALL_COLUMNS
    return df

def load_rul_labels(file_path: str) -> np.ndarray:
    """Loads ground truth RUL values for test engines."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"RUL labels file not found at: {file_path}")
        
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    return df.iloc[:, 0].values

if __name__ == "__main__":
    # Quick debug run
    print("Testing data loader...")
    try:
        train_df = load_dataset(Config.TRAIN_DATA_PATH)
        print(f"Loaded training data successfully. Shape: {train_df.shape}")
        print("First 2 rows:\n", train_df.head(2))
    except Exception as e:
        print("Error during test run:", e)
