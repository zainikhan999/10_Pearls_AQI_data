import hopsworks
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import os
API_KEY = os.environ["AQI_API"]
# --- Config ---
FG_NAME = "aqi_weather_features"
FG_VERSION = 1

LAT, LON = 33.5973, 73.0479
TIMEZONE = "Asia/Karachi"

# --- Login to Hopsworks ---
project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
fs = project.get_feature_store()
fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

# --- Get latest timestamp in feature store ---
fg_df = fg.read()
fg_df["time"] = pd.to_datetime(fg_df["time"], utc=True)
latest_time = fg_df["time"].max()

# Compute next hour in UTC
next_hour_utc = (latest_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

# Convert to date string for API
next_hour_date = next_hour_utc.date().strftime("%Y-%m-%d")

# --- AQI API ---
aqi_url = (
    f"https://air-quality-api.open-meteo.com/v1/air-quality?"
    f"latitude={LAT}&longitude={LON}&hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,"
    f"nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi&start_date={next_hour_date}&end_date={next_hour_date}&timezone=auto"
)

# --- Forecast Weather API ---
forecast_url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={LAT}&longitude={LON}&hourly=temperature_2m,relative_humidity_2m,rain,"
    f"wind_speed_10m,wind_direction_10m&timezone=auto"
)

def fetch_api_df(url, key="hourly"):
    response = requests.get(url).json()
    df = pd.DataFrame(response[key])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df

# Fetch data
aqi_df = fetch_api_df(aqi_url)
weather_df = fetch_api_df(forecast_url)

# Merge on time
merged = pd.merge(aqi_df, weather_df, on="time", how="inner")
merged = merged[merged["time"] == next_hour_utc]

# Insert only if data is found
if not merged.empty:
    fg.insert(merged, write_options={"wait_for_job": False})
    print(f"✅ Inserted new row for time: {next_hour_utc}")
else:
    print(f"⚠️ No data found for time {next_hour_utc}. Nothing inserted.")
