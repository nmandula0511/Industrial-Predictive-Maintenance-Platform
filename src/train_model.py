import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data (same as before)
columns = ["engine_id", "cycle"] + \
          [f"op_setting_{i}" for i in range(1, 4)] + \
          [f"sensor_{i}" for i in range(1, 22)]

df = pd.read_csv("data/train_FD001.txt", sep=" ", header=None)
df = df.dropna(axis=1)
df.columns = columns

# Create RUL
max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
max_cycle.columns = ["engine_id", "max_cycle"]
df = df.merge(max_cycle, on="engine_id")
df["RUL"] = df["max_cycle"] - df["cycle"]

# Feature selection (IMPORTANT)
features = [col for col in df.columns if "sensor" in col]

X = df[features]
y = df["RUL"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=50)
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Evaluate
import numpy as np

mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

print("\nModel Training Complete")
print("RMSE:", rmse)
# =========================
# FEATURE IMPORTANCE
# =========================

import pandas as pd

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(by="importance", ascending=False)

print("\nTop Important Features:")
print(feature_importance.head(10))
# =========================
# OEE CALCULATION (SIMULATED)
# =========================

# Availability (based on RUL threshold)
df["availability"] = df["RUL"].apply(lambda x: 1 if x > 30 else 0)

# Performance (normalized cycle)
df["performance"] = df["cycle"] / df["cycle"].max()

# Quality (simulate using stable sensors)
df["quality"] = (df["sensor_11"] / df["sensor_11"].max())

# OEE
df["OEE"] = df["availability"] * df["performance"] * df["quality"]

print("\nOEE Sample:")
print(df[["engine_id", "cycle", "RUL", "availability", "performance", "quality", "OEE"]].head(10))