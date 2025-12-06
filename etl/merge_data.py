#!/usr/bin/env python3
"""
Full ETL: Merge NOAA weather with USDA crop yields
- Production-ready with real station → county mapping using cKDTree
- Robust to missing counties / mismatched names
- Aggregates NOAA weather by county/year
- Saves clean CSV (without verbose date columns), aggregated CSV, and full SQLite DB
- Includes human-readable names for location and commodity
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import sqlite3


# -------------------------
# CONFIG
# -------------------------
NOAA_TOKEN = "ePeCMiiNqmbyLWOUhBNNLsyDkrmJaaIF"
CROP_FILE = "data/raw_usda/USDA_ALL_COMMODITIES_2005_2025.csv"
WEATHER_FILE = "data/raw_noaa/NOAA_ALL_2005_2025.csv"
STATION_CACHE_FILE = "data/raw_noaa/noaa_stations_full.csv"
COUNTY_CENTROIDS = "data/raw_usda/us_county_centroids.csv"
OUTPUT_CSV = "data/processed_noaa/merged_weather_crop_clean.csv"
AGG_OUTPUT_CSV = "data/processed_noaa/merged_weather_crop_aggregated.csv"
DB_PATH = "db/drought_yield.db"

NOAA_STATION_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/stations"
PAGE_LIMIT = 1000
RETRY_LIMIT = 5

# -------------------------
# FETCH NOAA STATIONS
# -------------------------
def fetch_noaa_stations_cached():
    if os.path.exists(STATION_CACHE_FILE):
        print("Loading stations from cache...")
        return pd.read_csv(STATION_CACHE_FILE)

    print("Downloading NOAA stations...")
    offset = 1
    stations = []

    while True:
        tries = 0
        while tries < RETRY_LIMIT:
            try:
                resp = requests.get(
                    NOAA_STATION_URL,
                    headers={"token": NOAA_TOKEN},
                    params={"limit": PAGE_LIMIT, "offset": offset},
                    timeout=30
                )
                if resp.status_code != 200:
                    tries += 1
                    time.sleep(2)
                    continue

                data = resp.json()
                results = data.get("results", [])
                stations.extend(results)
                print(f"Fetched {len(results)} stations at offset {offset}")

                if len(results) < PAGE_LIMIT:
                    df = pd.DataFrame(stations)
                    os.makedirs(os.path.dirname(STATION_CACHE_FILE), exist_ok=True)
                    df.to_csv(STATION_CACHE_FILE, index=False)
                    return df
                offset += PAGE_LIMIT
                break

            except Exception:
                tries += 1
                time.sleep(2)

        if tries >= RETRY_LIMIT:
            df = pd.DataFrame(stations)
            os.makedirs(os.path.dirname(STATION_CACHE_FILE), exist_ok=True)
            df.to_csv(STATION_CACHE_FILE, index=False)
            return df

# -------------------------
# LOAD CSVs
# -------------------------
print("Loading USDA crop, NOAA weather, and county centroids CSVs...")
crop_df = pd.read_csv(CROP_FILE)
weather_df = pd.read_csv(WEATHER_FILE)
stations_df = fetch_noaa_stations_cached().dropna(subset=["latitude","longitude"]).reset_index(drop=True)
centroids_df = pd.read_csv(COUNTY_CENTROIDS)

centroids_df = centroids_df[centroids_df['state'].astype(str).str.upper().str.strip().isin(['KENTUCKY', 'KY'])].reset_index(drop=True)


# -------------------------
# FILTER NOAA STATIONS TO KENTUCKY ONLY (robust)
# -------------------------
print("Filtering NOAA stations to Kentucky only...")

# Keep only stations whose name ends with ', KY US' (strong guarantee)
stations_df = stations_df[stations_df['name'].str.upper().str.endswith(', KY US')].reset_index(drop=True)

# Build KDTree only on Kentucky county centroids
ky_centroids = centroids_df.copy()
ky_coords = ky_centroids[['latitude', 'longitude']].to_numpy()

station_coords = stations_df[['latitude', 'longitude']].to_numpy()
tree = cKDTree(ky_coords)

# Map stations to nearest KY county using KDTree
dist, idx = tree.query(station_coords, k=1)
stations_df['nearest_ky_fips'] = ky_centroids.iloc[idx]['cfips'].values
stations_df['distance_to_ky'] = dist

# Optional: further filter by distance to KY county if needed
stations_df = stations_df[stations_df['distance_to_ky'] < 0.5].reset_index(drop=True)

print(f"Stations kept after KY filtering: {len(stations_df)}")

# -------------------------
# KY-ONLY FILTER (minimal changes)
# -------------------------
# Keep everything else identical — restrict centroids and crop data to Kentucky only.
# Accept both full state name and postal code variants.
crop_df = crop_df[crop_df['state_name'].astype(str).str.upper().str.strip().isin(['KENTUCKY', 'KY'])].reset_index(drop=True)
#centroids_df = centroids_df[centroids_df['state'].astype(str).str.upper().str.strip().isin(['KENTUCKY', 'KY'])].reset_index(drop=True)

# -------------------------
# DIMENSION TABLES
# -------------------------
# -------------------------
# CREATE DIM_STATION TABLE
# -------------------------
dim_station = stations_df.rename(columns={
    "id": "station_id",
    "name": "name",
    "latitude": "latitude",
    "longitude": "longitude",
    "elevation": "elevation",
    "elevationUnit": "elevation_unit",
    "mindate": "mindate",
    "maxdate": "maxdate",
    "datacoverage": "datacoverage"
})[["station_id", "name", "latitude", "longitude", "elevation",
    "elevation_unit", "mindate", "maxdate", "datacoverage"]]

print(f"dim_station rows: {len(dim_station)}")
print(dim_station.head())
# 2. dim_date
noaa_dates = pd.to_datetime(weather_df['date']).dt.date
usda_years = pd.to_datetime(crop_df['year'].astype(str) + "-01-01").dt.date
all_dates = pd.Series(pd.concat([noaa_dates, usda_years]).unique(), name="date")
dim_date = pd.DataFrame(all_dates)
dim_date['year'] = pd.DatetimeIndex(dim_date['date']).year
dim_date.reset_index(inplace=True)
dim_date.rename(columns={"index":"date_id"}, inplace=True)

# 3. dim_location
dim_location = centroids_df.rename(columns={
    "state":"state_name",
    "county":"county_name",
    "cfips":"county_fips",
    "latitude":"lat",
    "longitude":"lon"
}).copy()
dim_location['county_name'] = dim_location['county_name'].str.upper().str.replace(r'\s+COUNTY$', '', regex=True)
dim_location['state_name'] = dim_location['state_name'].str.upper()
dim_location['location_id'] = range(1, len(dim_location)+1)

# 4. dim_commodity
dim_commodity = crop_df[['commodity_desc','group_desc','class_desc','prodn_practice_desc']].drop_duplicates().reset_index(drop=True)
dim_commodity['commodity_id'] = range(1, len(dim_commodity)+1)

# -------------------------
# STATION → LOCATION MAPPING USING KDTree
# -------------------------
print("Mapping stations to nearest county centroids...")
station_coords = dim_station[['latitude','longitude']].to_numpy()
county_coords = dim_location[['lat','lon']].to_numpy()
tree = cKDTree(county_coords)
_, indices = tree.query(station_coords, k=1)

station_to_location = pd.DataFrame({
    'station_id': dim_station['station_id'],
    'location_id': dim_location.iloc[indices]['location_id'].values
})

# -------------------------
# FACT TABLES
# -------------------------
fact_yield = crop_df.copy()
fact_yield['county_name'] = fact_yield['county_name'].str.upper().str.replace(r'\s+COUNTY$', '', regex=True)
fact_yield['state_name'] = fact_yield['state_name'].str.upper()

# mappings for exact matches
loc_map = dict(zip(zip(dim_location['county_name'], dim_location['state_name']), dim_location['location_id']))
commodity_map = dict(zip(zip(dim_commodity['commodity_desc'], dim_commodity['prodn_practice_desc']), dim_commodity['commodity_id']))

# KDTree fallback
county_tree = cKDTree(dim_location[['lat','lon']].to_numpy())
county_ids = dim_location['location_id'].values

def get_location_id(row):
    key = (row['county_name'], row['state_name'])
    if key in loc_map:
        return loc_map[key]
    lat = row.get('latitude', np.nan)
    lon = row.get('longitude', np.nan)
    if pd.isna(lat) or pd.isna(lon):
        state_centroids = dim_location[dim_location['state_name'] == row['state_name']]
        if not state_centroids.empty:
            lat, lon = state_centroids[['lat','lon']].mean().values
        else:
            lat, lon = dim_location[['lat','lon']].mean().values
    _, idx = county_tree.query([lat, lon])
    return county_ids[idx]

fact_yield['location_id'] = fact_yield.apply(get_location_id, axis=1)
fact_yield['commodity_id'] = fact_yield.apply(lambda x: commodity_map.get((x['commodity_desc'], x['prodn_practice_desc']), -1), axis=1)
fact_yield['date_id'] = pd.to_datetime(fact_yield['year'].astype(str)+"-01-01").map(dict(zip(dim_date['date'], dim_date['date_id'])))
fact_yield = fact_yield[['location_id','commodity_id','date_id','Value','unit_desc','sector_desc','group_desc','region_desc']]
fact_yield.rename(columns={"Value":"value"}, inplace=True)
fact_yield['yield_id'] = range(1, len(fact_yield)+1)

# -------------------------
# fact_weather
# -------------------------
date_map = dict(zip(dim_date['date'], dim_date['date_id']))
weather_df['date_id'] = pd.to_datetime(weather_df['date']).dt.date.map(date_map)
fact_weather = weather_df.rename(columns={"station":"station_id","value":"value","attributes":"attributes"})[["station_id","date_id","datatype","value","attributes"]]
fact_weather['weather_id'] = range(1, len(fact_weather)+1)

# -------------------------
# AGGREGATE WEATHER BY COUNTY/YEAR
# -------------------------
fact_weather_county = fact_weather.merge(station_to_location, on='station_id')
fact_weather_county = fact_weather_county.merge(dim_date[['date_id','year']], on='date_id')
weather_agg = fact_weather_county.groupby(['location_id','year','datatype'])['value'].agg(['sum','mean']).reset_index()
weather_agg_pivot = weather_agg.pivot_table(index=['location_id','year'], columns='datatype', values=['sum','mean'])
weather_agg_pivot.columns = ['_'.join(col).strip() for col in weather_agg_pivot.columns.values]
weather_agg_pivot.reset_index(inplace=True)

# -------------------------
# FINAL MERGED TABLE WITH NAMES
# -------------------------
fact_yield = fact_yield.merge(dim_date[['date_id','year']], on='date_id')
final_merged = fact_yield.merge(weather_agg_pivot, on=['location_id','year'], how='left')

# Merge human-readable names
final_with_names = final_merged.merge(
    dim_location[['location_id', 'county_name', 'state_name']],
    on='location_id', how='left'
).merge(
    dim_commodity[['commodity_id','commodity_desc','prodn_practice_desc']],
    on='commodity_id', how='left'
)

# Ensure yield_id is kept
final_with_names['yield_id'] = final_merged['yield_id']

# -------------------------
# SAVE CLEAN DETAILED CSV
# -------------------------
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
csv_cols = ['yield_id', 'location_id', 'state_name', 'county_name',
            'commodity_id', 'commodity_desc', 'prodn_practice_desc',
            'value','unit_desc','sector_desc','group_desc','region_desc'] + \
           [c for c in final_with_names.columns if c.startswith(('sum_','mean_'))]

final_with_names[csv_cols].to_csv(OUTPUT_CSV, index=False)
print(f"✅ Clean detailed CSV saved to {OUTPUT_CSV}")

# -------------------------
# AGGREGATE FOR COUNTY/YEAR/COMMODITY
# -------------------------
agg_county_commodity = final_with_names.groupby(
    ['location_id', 'year', 'commodity_id', 'state_name', 'county_name', 'commodity_desc', 'prodn_practice_desc'],
    as_index=False
).agg({
    'value': 'mean',
    'unit_desc': 'first',
    'sector_desc': 'first',
    'group_desc': 'first',
    'region_desc': 'first',
    **{col: 'mean' for col in final_with_names.columns if col.startswith(('sum_','mean_'))}
})

# -------------------------
# CLEAN AGGREGATED (NO DUPES, NO MISSING WEATHER)
# -------------------------
print("Cleaning aggregated dataset...")

# Identify weather columns (mean/sum)
weather_cols = [c for c in agg_county_commodity.columns if c.startswith(("mean_", "sum_"))]

# If there are weather columns, require them all to be non-missing; otherwise just dedupe and filter yields
if weather_cols:
    agg_clean = agg_county_commodity.dropna(subset=weather_cols)
else:
    agg_clean = agg_county_commodity.copy()

# Remove duplicate county-year-commodity records (keep first occurrence)
agg_clean = agg_clean.drop_duplicates(subset=["location_id", "year", "commodity_id"], keep="first")

# Remove zero or negative yields (sensible filter for modeling)
if "value" in agg_clean.columns:
    agg_clean = agg_clean[agg_clean["value"].notna() & (agg_clean["value"] > 0)]

print(f"Rows before cleaning: {len(agg_county_commodity)}")
print(f"Rows after cleaning: {len(agg_clean)}")

# -------------------------
# SAVE AGGREGATED CSV (CLEAN) 
# -------------------------
os.makedirs(os.path.dirname(AGG_OUTPUT_CSV), exist_ok=True)
agg_clean.to_csv(AGG_OUTPUT_CSV, index=False)
print(f" Clean aggregated CSV saved to {AGG_OUTPUT_CSV}")

OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dimension tables
dim_station.to_csv(f"{OUTPUT_DIR}/dim_station.csv", index=False)
dim_date.to_csv(f"{OUTPUT_DIR}/dim_date.csv", index=False)
dim_location.to_csv(f"{OUTPUT_DIR}/dim_location.csv", index=False)
dim_commodity.to_csv(f"{OUTPUT_DIR}/dim_commodity.csv", index=False)

# Bridge table
station_to_location.to_csv(f"{OUTPUT_DIR}/station_to_location.csv", index=False)

# Fact tables
fact_weather.to_csv(f"{OUTPUT_DIR}/fact_weather.csv", index=False)
fact_yield.to_csv(f"{OUTPUT_DIR}/fact_yield.csv", index=False)

print(f"✅ All 7 ERD tables exported to: {OUTPUT_DIR}")

# -------------------------
# SAVE FULL SQLITE DB (CLEAN AGG INSERT)
# -------------------------
conn = sqlite3.connect(DB_PATH)

dim_station.to_sql("dim_station", conn, if_exists="replace", index=False)
dim_date.to_sql("dim_date", conn, if_exists="replace", index=False)
dim_location.to_sql("dim_location", conn, if_exists="replace", index=False)
dim_commodity.to_sql("dim_commodity", conn, if_exists="replace", index=False)
station_to_location.to_sql("station_to_location", conn, if_exists="replace", index=False)
fact_weather.to_sql("fact_weather", conn, if_exists="replace", index=False)
fact_yield.to_sql("fact_yield", conn, if_exists="replace", index=False)
final_with_names.to_sql("merged_weather_yield", conn, if_exists="replace", index=False)

# write clean aggregated table (agg_clean) to DB
agg_clean.to_sql("merged_weather_county_commodity", conn, if_exists="replace", index=False)
print(" Clean aggregated table written to database")

conn.close()
print(f" SQLite database created/overwritten at {DB_PATH}")
