#!/usr/bin/env python3
"""
Train Linear Regression model for crop yield prediction (Drought Yield Analysis)
Features: weather aggregates + crop one-hot
Target: yield (value)
Temporal split:
    - Train/Val: years < 2021 (80/20)
    - Test: years 2021–2024
Saves: model + OneHotEncoder + feature columns (pickle), metrics (JSON)
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json
import os

DB_PATH = "db/drought_yield.db"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "linear_yield_model.pkl")
FEATURE_FILE = os.path.join(MODEL_DIR, "feature_columns.json")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# LOAD DATA FROM SQLITE
# -------------------------
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM merged_weather_county_commodity", conn)
conn.close()

# -------------------------
# Split by temporal logic
# -------------------------
df_trainval = df[df['year'] < 2021].copy()
df_test = df[df['year'].between(2021, 2024)].copy()

# -------------------------
# FEATURES & TARGET
# -------------------------
weather_cols = [c for c in df.columns if c.startswith(("mean_", "sum_"))]
categorical_cols = ["commodity_desc", "prodn_practice_desc"]
target_col = "value"

# Drop missing data
df_trainval = df_trainval.dropna(subset=weather_cols + [target_col])
df_test = df_test.dropna(subset=weather_cols + [target_col])

# -------------------------
# One-hot encode categorical columns
# -------------------------
ohe = OneHotEncoder(sparse_output=False, drop='first')
ohe.fit(df_trainval[categorical_cols])

X_trainval_cat = ohe.transform(df_trainval[categorical_cols])
X_test_cat = ohe.transform(df_test[categorical_cols])

# Combine weather + categorical features
X_trainval = np.hstack([df_trainval[weather_cols].values, X_trainval_cat])
X_test = np.hstack([df_test[weather_cols].values, X_test_cat])
y_trainval = df_trainval[target_col].values
y_test = df_test[target_col].values

# Save feature names
feature_cols = list(weather_cols) + list(ohe.get_feature_names_out(categorical_cols))
with open(FEATURE_FILE, "w") as f:
    json.dump(feature_cols, f)

# -------------------------
# Train/validation split (random, 80/20)
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

# -------------------------
# Train Linear Regression
# -------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Save everything together
with open(MODEL_FILE, "wb") as f:
    pickle.dump({
        "model": model,
        "ohe": ohe,
        "feature_cols": feature_cols
    }, f)

# -------------------------
# Evaluate
# -------------------------
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

metrics = {
    "train_r2": r2_score(y_train, y_train_pred),
    "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
    "val_r2": r2_score(y_val, y_val_pred),
    "val_rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
    "test_r2": r2_score(y_test, y_test_pred),
    "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred))
}

# Save metrics
with open(METRICS_FILE, "w") as f:
    json.dump(metrics, f)

print("✅ Model trained and saved at", MODEL_FILE)
print("Model Performance:")
for k,v in metrics.items():
    print(f"{k}: {v:.3f}")
