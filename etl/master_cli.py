#!/usr/bin/env python3
"""
Master CLI for Drought-Yield Project
- Run NOAA ETL
- Run USDA ETL
- Merge datasets
- Train Yield Prediction Model
- Predict Yield Interactively
- Query merged SQLite database (cascading)
"""

import subprocess
import pandas as pd
import sqlite3
import os

def menu(options):
    print("\nSelect an option:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print("  0. Exit")
    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit():
            choice = int(choice)
            if 0 <= choice <= len(options):
                return choice
        print("Invalid choice. Please try again.")

# -------------------------
# ETL Functions
# -------------------------
def run_noaa_etl():
    print("Running NOAA ETL...")
    subprocess.run(["python", "scripts/etl/noaa_etl.py"], check=True)

def run_usda_etl():
    print("Running USDA ETL...")
    subprocess.run(["python", "scripts/etl/usda_etl.py"], check=True)

def merge_datasets():
    print("Running merge_data.py ETL...")
    subprocess.run(["python", "scripts/etl/merge_data.py"], check=True)

# -------------------------
# Query Merged Dataset (Cascading)
# -------------------------
def query_merged():
    db_path = "db/drought_yield.db"
    if not os.path.exists(db_path):
        print("⚠ Database not found. Please merge datasets first.")
        return

    conn = sqlite3.connect(db_path)
    print("\nQuery merged dataset (cascading selection)")

    # Step 1: Crop
    all_crops = pd.read_sql_query("SELECT DISTINCT commodity_desc FROM merged_weather_county_commodity", conn)
    crop = input(f"Enter crop (available: {', '.join(all_crops['commodity_desc'].unique())}): ").strip().upper()
    df_filtered = pd.read_sql_query(
        "SELECT * FROM merged_weather_county_commodity WHERE UPPER(commodity_desc) = :crop",
        conn, params={"crop": crop}
    )
    if df_filtered.empty:
        print(f"No data found for crop '{crop}'.")
        conn.close()
        return

    # Step 2: State
    all_states = df_filtered['state_name'].dropna().unique()
    state = input(f"Enter state (available: {', '.join(all_states)}): ").strip().upper()
    df_filtered = df_filtered[df_filtered['state_name'].str.upper() == state]
    if df_filtered.empty:
        print(f"No data found for crop '{crop}' in state '{state}'.")
        conn.close()
        return

    # Step 3: County
    all_counties = df_filtered['county_name'].dropna().unique()
    county = input(f"Enter county (available: {', '.join(all_counties)}): ").strip().upper()
    df_filtered = df_filtered[df_filtered['county_name'].str.upper() == county]
    if df_filtered.empty:
        print(f"No data found for crop '{crop}' in state '{state}', county '{county}'.")
        conn.close()
        return

    # Step 4: Year range
    min_year = int(df_filtered['year'].min())
    max_year = int(df_filtered['year'].max())
    print(f"Data available from {min_year} to {max_year}.")
    while True:
        year_start = input(f"Start year [{min_year}-{max_year}]: ").strip()
        year_end = input(f"End year [{min_year}-{max_year}]: ").strip()
        try:
            year_start = int(year_start)
            year_end = int(year_end)
            if min_year <= year_start <= max_year and min_year <= year_end <= max_year and year_start <= year_end:
                break
            else:
                print("⚠ Years must be within available range and start <= end.")
        except ValueError:
            print("⚠ Please enter valid years.")

    df_filtered = df_filtered[(df_filtered['year'] >= year_start) & (df_filtered['year'] <= year_end)]

    # Show results
    print(f"\nFound {len(df_filtered)} records for {crop} / {state} / {county} from {year_start} to {year_end}")
    print(df_filtered.head(10).to_string(index=False))

    export = input("Export results to CSV? (y/n): ").strip().lower()
    if export == 'y':
        path = input("Enter CSV filename (e.g., results.csv): ").strip()
        df_filtered.to_csv(path, index=False)
        print(f"Results exported to {path}")

    conn.close()

# -------------------------
# Train Model
# -------------------------
def train_model():
    print("Training yield prediction model...")
    subprocess.run(["python", "scripts/etl/train_model.py"], check=True)
    print("Model training complete.")

# -------------------------
# Predict Yield Interactively
# -------------------------
def predict_yield():
    print("Starting interactive yield prediction...")
    subprocess.run(["python", "scripts/etl/predict_yield_risk.py"], check=True)

# -------------------------
# Main CLI Loop
# -------------------------
def main():
    options = [
        "Fetch or load NOAA weather data",
        "Fetch or load USDA crop data",
        "Merge datasets",
        "Train yield prediction model",
        "Predict crop yield interactively",
        "Query merged dataset"
    ]

    while True:
        choice = menu(options)
        if choice == 0:
            break
        elif choice == 1:
            run_noaa_etl()
        elif choice == 2:
            run_usda_etl()
        elif choice == 3:
            merge_datasets()
        elif choice == 4:
            train_model()
        elif choice == 5:
            predict_yield()
        elif choice == 6:
            query_merged()

    print("Exiting master interface.")

if __name__ == "__main__":
    print("Starting Drought-Yield Master CLI")
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
