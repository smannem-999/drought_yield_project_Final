# usda_etl.py
import requests
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------
# Configuration
# -------------------------------
API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
API_KEY = "AABFAE72-27BB-35B1-9B8A-FA480E33AA06"

RAW_USDA_DIR = "data/raw_usda"
os.makedirs(RAW_USDA_DIR, exist_ok=True)

COMMODITIES = ["CORN", "SOYBEANS", "WHEAT", "SORGHUM", "BARLEY"]
START_YEAR = 2005
END_YEAR = 2025
DELAY_BETWEEN_REQUESTS = 1      # seconds per request
MAX_WORKERS = 10                # number of parallel threads

MASTER_CSV = os.path.join(
    RAW_USDA_DIR, 
    f"USDA_ALL_COMMODITIES_{START_YEAR}_{END_YEAR}.csv"
)


# -------------------------------
# Fetch data for a single commodity/year (Kentucky ONLY)
# -------------------------------
def fetch_usda_data(commodity, year):
    params = {
        "key": API_KEY,
        "commodity_desc": commodity,
        "year__GE": str(year),
        "year__LE": str(year),
        "agg_level_desc": "COUNTY",
        "state_alpha": "KY",                    # ← ONLY Kentucky, all counties
        "statisticcat_desc": "YIELD",
        "format": "JSON"
    }

    try:
        response = requests.get(API_URL, params=params)
        time.sleep(DELAY_BETWEEN_REQUESTS)

        if response.status_code != 200:
            print(f"[{commodity} {year}] Error: Status code {response.status_code}")
            return None

        data = response.json().get("data", [])
        if not data:
            print(f"[{commodity} {year}] No data returned")
            return None

        df = pd.DataFrame(data)

        # Save each year individually
        df.to_csv(
            os.path.join(RAW_USDA_DIR, f"{commodity}_{year}.csv"), 
            index=False
        )

        return df

    except Exception as e:
        print(f"[{commodity} {year}] Request failed: {e}")
        return None


# -------------------------------
# Create final master CSV (all commodities)
# -------------------------------
def create_master_csv():
    all_files = [
        os.path.join(RAW_USDA_DIR, f)
        for f in os.listdir(RAW_USDA_DIR)
        if f.endswith(f"_ALL_{START_YEAR}_{END_YEAR}.csv")
    ]

    if not all_files:
        print("No per-commodity files found to build master CSV.")
        return

    combined = []
    for file_path in all_files:
        commodity = os.path.basename(file_path).split("_ALL_")[0]
        df = pd.read_csv(file_path)
        df["COMMODITY"] = commodity
        combined.append(df)

    master_df = pd.concat(combined, ignore_index=True)
    master_df.to_csv(MASTER_CSV, index=False)
    print(f"\nMaster CSV saved to: {MASTER_CSV}")


# -------------------------------
# Run ETL workflow with progress
# -------------------------------
def run_usda_etl():
    # Prepare task list
    tasks = [
        (commodity, year)
        for commodity in COMMODITIES
        for year in range(START_YEAR, END_YEAR + 1)
    ]

    total_tasks = len(tasks)
    completed = 0

    # Store all successfully returned DataFrames
    all_results = {c: [] for c in COMMODITIES}

    print(f"Starting ETL: {total_tasks} total fetch operations...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_usda_data, commodity, year): (commodity, year)
            for commodity, year in tasks
        }

        for future in as_completed(futures):
            commodity, year = futures[future]
            result = future.result()
            completed += 1

            pct = (completed / total_tasks) * 100
            print(
                f"[{completed}/{total_tasks}] "
                f"{pct:5.1f}% complete – {commodity} {year}"
            )

            if result is not None:
                all_results[commodity].append(result)

    print("\nCombining yearly data per commodity...")

    for commodity, dfs in all_results.items():
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            path = os.path.join(
                RAW_USDA_DIR, 
                f"{commodity}_ALL_{START_YEAR}_{END_YEAR}.csv"
            )
            combined.to_csv(path, index=False)
            print(f"  ✔ {commodity}: saved {len(combined)} rows")
        else:
            print(f"  ✖ {commodity}: no data fetched")

    print("\nBuilding master file across all commodities...")
    create_master_csv()

    print("\nETL complete.")


# -------------------------------
# Run if executed directly
# -------------------------------
if __name__ == "__main__":
    run_usda_etl()
