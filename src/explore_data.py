import pandas as pd

# Column names
columns = ["engine_id", "cycle"] + \
          [f"op_setting_{i}" for i in range(1, 4)] + \
          [f"sensor_{i}" for i in range(1, 22)]

# Load training data
train_df = pd.read_csv(
    "data/train_FD001.txt",
    sep=" ",
    header=None
)

# Remove empty columns
train_df = train_df.dropna(axis=1)

# Assign column names
train_df.columns = columns

# Show first rows
print("First 5 rows:")
print(train_df.head())

print("\nShape of dataset:", train_df.shape)

print("\nColumns:")
print(train_df.columns)

# =========================
# CREATE RUL (IMPORTANT)
# =========================

# Max cycle per engine
max_cycle = train_df.groupby("engine_id")["cycle"].max().reset_index()
max_cycle.columns = ["engine_id", "max_cycle"]

# Merge
train_df = train_df.merge(max_cycle, on="engine_id")

# RUL calculation
train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]

# Show RUL
print("\nRUL added:")
print(train_df[["engine_id", "cycle", "max_cycle", "RUL"]].head())
# =========================
# FEATURE ENGINEERING
# =========================

# Sort data (VERY IMPORTANT)
train_df = train_df.sort_values(by=["engine_id", "cycle"])

# Create rolling mean for sensor_1 (example)
train_df["sensor_1_ma"] = train_df.groupby("engine_id")["sensor_1"].rolling(window=5).mean().reset_index(level=0, drop=True)

# Fill missing values
train_df["sensor_1_ma"] = train_df["sensor_1_ma"].bfill()

# Show new feature
print("\nFeature Engineering:")
print(train_df[["engine_id", "cycle", "sensor_1", "sensor_1_ma"]].head(10))
# =========================
# FEATURE ENGINEERING
# =========================

# Sort data
train_df = train_df.sort_values(by=["engine_id", "cycle"])

# Rolling mean (moving average)
train_df["sensor_1_ma"] = (
    train_df.groupby("engine_id")["sensor_1"]
    .rolling(window=5)
    .mean()
    .reset_index(level=0, drop=True)
)

# Fill missing values
train_df["sensor_1_ma"] = train_df["sensor_1_ma"].bfill()

# Show output
print("\nFeature Engineering Output:")
print(train_df[["engine_id", "cycle", "sensor_1", "sensor_1_ma"]].head(10))