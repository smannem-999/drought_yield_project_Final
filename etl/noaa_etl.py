# noaa_etl.py
import requests
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------
# Configuration
# -------------------------------
BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
TOKEN = "ePeCMiiNqmbyLWOUhBNNLsyDkrmJaaIF"

RAW_NOAA_DIR = "data/raw_noaa"
os.makedirs(RAW_NOAA_DIR, exist_ok=True)

DATASET_ID = "GSOY"  # Global Summary of the Year
DATA_TYPES = ["PRCP", "TAVG", "TMAX", "TMIN"]  # Precipitation & temperature
START_YEAR = 2005
END_YEAR = 2025
DELAY_BETWEEN_REQUESTS = 1  # seconds
MAX_WORKERS = 5  # concurrency

# -------------------------------
# Helper function to fetch NOAA data for a given year
# -------------------------------
def fetch_noaa_year(year):
    headers = {"token": TOKEN}
    all_records = []

    # NOAA API limits: use limit and offset for pagination
    limit = 1000
    offset = 1

    while True:
        params = {
            "datasetid": DATASET_ID,
            "datatypeid": ",".join(DATA_TYPES),
            "startdate": f"{year}-01-01",
            "enddate": f"{year}-12-31",
            "units": "standard",
            "limit": limit,
            "offset": offset,
        }

        try:
            response = requests.get(BASE_URL, headers=headers, params=params)
            if response.status_code != 200:
                print(f"[{year}] Error: Status code {response.status_code}")
                break

            data = response.json().get("results", [])
            if not data:
                break

            all_records.extend(data)
            offset += limit
            time.sleep(DELAY_BETWEEN_REQUESTS)

        except requests.exceptions.RequestException as e:
            print(f"[{year}] Request failed: {e}")
            break
        except ValueError as e:
            print(f"[{year}] JSON decode error: {e}")
            break

    if all_records:
        df = pd.DataFrame(all_records)
        csv_path = os.path.join(RAW_NOAA_DIR, f"NOAA_{year}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[{year}] Saved {len(df)} records to {csv_path}")
        return df
    else:
        print(f"[{year}] No records fetched")
        return None

# -------------------------------
# Main ETL function
# -------------------------------
def run_noaa_etl():  # <- new function to call from master script
    all_data = []

    years = list(range(START_YEAR, END_YEAR + 1))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_year = {executor.submit(fetch_noaa_year, year): year for year in years}

        for future in as_completed(future_to_year):
            year = future_to_year[future]
            df_year = future.result()
            if df_year is not None:
                all_data.append(df_year)

    # Combine all years into a single CSV
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_csv_path = os.path.join(RAW_NOAA_DIR, f"NOAA_ALL_{START_YEAR}_{END_YEAR}.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"All NOAA data combined and saved to: {combined_csv_path}")
    else:
        print("No NOAA data fetched.")

# Keep the script runnable directly
if __name__ == "__main__":
    run_noaa_etl()
