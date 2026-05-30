import os

class Config:
    # Workspace directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
    
    # Ensure directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Dataset Files
    TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_FD001.txt")
    TEST_DATA_PATH = os.path.join(DATA_DIR, "test_FD001.txt")
    RUL_DATA_PATH = os.path.join(DATA_DIR, "RUL_FD001.txt")
    MAINTENANCE_MANUAL_PATH = os.path.join(DATA_DIR, "maintenance_manual.md")
    
    # Model Artifacts
    MODEL_PATH = os.path.join(MODELS_DIR, "predictor.pkl")
    SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
    FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "feature_cols.json")
    
    # CMAPSS Data Columns
    ID_COLS = ["engine_id", "cycle"]
    OP_SETTINGS = [f"op_setting_{i}" for i in range(1, 4)]
    SENSORS = [f"sensor_{i}" for i in range(1, 22)]
    ALL_COLUMNS = ID_COLS + OP_SETTINGS + SENSORS
    
    # Selected sensor columns based on feature importance
    SELECTED_SENSORS = [
        "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", 
        "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14", 
        "sensor_15", "sensor_17", "sensor_20", "sensor_21"
    ]
    
    # Model Training Parameters
    N_ESTIMATORS = 100
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # OEE Metrics Constants
    RUL_CRITICAL_THRESHOLD = 30  # Downtime risk (Availability)
    OEE_BASELINE_CYCLE = 300     # Max cycle expectation (Performance scale)
    SENSOR_11_STABLE_MAX = 525.0  # Quality check divisor
