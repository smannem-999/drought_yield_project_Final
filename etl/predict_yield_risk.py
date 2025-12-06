#!/usr/bin/env python3
"""
Generate predictive CSV for Power BI:
- Historical predictions (existing years)
- Future predictions (2025 next season)
"""

import pandas as pd
import numpy as np
import sqlite3
import joblib
import os

# -------------------------
# Paths
# -------------------------
DB_PATH = "db/drought_yield.db"
MODEL_FILE = "models/linear_yield_model.pkl"
OUTPUT_CSV = "full_powerbi_predictive_dataset.csv"

# -------------------------
# Load model
# -------------------------
saved = joblib.load(MODEL_FILE)
model = saved["model"]
ohe = saved["ohe"]

# -------------------------
# Load base dataset
# -------------------------
conn = sqlite3.connect(DB_PATH)
df_hist = pd.read_sql("SELECT * FROM merged_weather_county_commodity", conn)
conn.close()

# Keep required columns
weather_cols = [c for c in df_hist.columns if c.startswith(("mean_", "sum_"))]
cat_cols = ["commodity_desc", "prodn_practice_desc"]

df_hist = df_hist.dropna(subset=weather_cols + cat_cols + ["value"]).copy()

# -------------------------
# Create future-year dataset (2025)
# -------------------------
numeric_cols = df_hist.select_dtypes(include=["number"]).columns

df_future = (
    df_hist
    .groupby(["state_name", "county_name", "commodity_desc", "prodn_practice_desc"])[numeric_cols]
    .mean()
    .reset_index()
)
df_future["year"] = 2025

# -------------------------
# Scenario adjustments
# -------------------------
scenarios = {
    "LOW_RAIN": 0.80,
    "NORMAL": 1.00,
    "HIGH_RAIN": 1.20
}

future_rows = []

for scen_name, factor in scenarios.items():
    temp = df_future.copy()
    temp["scenario"] = scen_name

    # Adjust precipitation + temperature
    temp["mean_PRCP"] = temp["mean_PRCP"] * factor
    temp["sum_PRCP"] = temp["sum_PRCP"] * factor
    temp["mean_TAVG"] = temp["mean_TAVG"] * (0.98 if factor < 1 else 1.02)
    temp["sum_TAVG"] = temp["sum_TAVG"] * (0.98 if factor < 1 else 1.02)

    future_rows.append(temp)

df_scenarios = pd.concat(future_rows, ignore_index=True)

# -------------------------
# Predict yields
# -------------------------
def predict(df):
    X_num = df[weather_cols].values
    X_cat = ohe.transform(df[cat_cols])
    preds = model.predict(np.hstack([X_num, X_cat])).clip(min=0)
    return preds

df_hist["predicted_yield"] = predict(df_hist)
df_scenarios["predicted_yield"] = predict(df_scenarios)

# -------------------------
# Historical median
# -------------------------
med = df_hist.groupby(["state_name","county_name","commodity_desc"])["value"].median().reset_index()
med.rename(columns={"value": "historical_median_yield"}, inplace=True)

df_hist = df_hist.merge(med, on=["state_name","county_name","commodity_desc"], how="left")
df_scenarios = df_scenarios.merge(med, on=["state_name","county_name","commodity_desc"], how="left")

# -------------------------
# Risk metric
# -------------------------
for df in [df_hist, df_scenarios]:
    df["yield_vs_historical"] = df["predicted_yield"] - df["historical_median_yield"]
    df["yield_drop_pct"] = ((df["historical_median_yield"] - df["predicted_yield"]) / 
                            df["historical_median_yield"] * 100).round(2)

# -------------------------
# Combine and export
# -------------------------
final = pd.concat([df_hist, df_scenarios], ignore_index=True)
final.to_csv(OUTPUT_CSV, index=False)

print("\n✅ Predictive CSV created:")
print(OUTPUT_CSV)
print("\nColumns:")
print(final.columns.tolist())
