# import os
# import hopsworks
# import pandas as pd
# import requests
# from datetime import datetime, timedelta, timezone
# from zoneinfo import ZoneInfo

# # --- Config ---
# API_KEY = os.environ["HOPSWORKS_API_KEY"]
# FG_NAME = "aqi_weather_features"
# FG_VERSION = 1

# LAT, LON = 33.5973, 73.0479
# TIMEZONE = "Asia/Karachi"

# # --- Login to Hopsworks ---
# project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
# fs = project.get_feature_store()
# fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

# # --- Get latest timestamp in feature store ---
# fg_df = fg.read()
# fg_df["time"] = pd.to_datetime(fg_df["time"], utc=True)
# latest_time = fg_df["time"].max()

# # --- Compute next hour in UTC ---
# next_hour_utc = (latest_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
# next_hour_date = next_hour_utc.date().strftime("%Y-%m-%d")

# # --- AQI API URL ---
# aqi_url = (
#     f"https://air-quality-api.open-meteo.com/v1/air-quality?"
#     f"latitude={LAT}&longitude={LON}&hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,"
#     f"nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi&start_date={next_hour_date}&end_date={next_hour_date}&timezone=auto"
# )

# # --- Forecast Weather API URL ---
# forecast_url = (
#     f"https://api.open-meteo.com/v1/forecast?"
#     f"latitude={LAT}&longitude={LON}&hourly=temperature_2m,relative_humidity_2m,rain,"
#     f"wind_speed_10m,wind_direction_10m&timezone=auto&past_days=1"
# )

# def fetch_api_df(url, key="hourly", is_local=False):
#     response = requests.get(url).json()
#     df = pd.DataFrame(response[key])
#     if is_local:
#         df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("Asia/Karachi").dt.tz_convert("UTC")
#     else:
#         df["time"] = pd.to_datetime(df["time"], utc=True)
#     return df

# # --- Fetch API Data ---
# aqi_df = fetch_api_df(aqi_url)                     # already in UTC
# weather_df = fetch_api_df(forecast_url, is_local=True)  # local → convert to UTC

# print("Available AQI timestamps:", aqi_df["time"].dt.strftime('%Y-%m-%d %H:%M:%S').tolist())
# print("Available Weather timestamps:", weather_df["time"].dt.strftime('%Y-%m-%d %H:%M:%S').tolist())
# print("🕓 Looking for:", next_hour_utc)

# # --- Merge and Filter ---
# merged = pd.merge(aqi_df, weather_df, on="time", how="inner")
# merged = merged[merged["time"] == next_hour_utc]

# # --- Convert types to match schema ---
# float_columns = [
#     "carbon_dioxide", "us_aqi",
#     "relative_humidity_2m", "wind_direction_10m"
# ]
# for col in float_columns:
#     if col in merged.columns:
#         merged[col] = merged[col].astype(float)

# # --- Insert to Feature Store ---
# if not merged.empty:
#     fg.insert(merged, write_options={"wait_for_job": False})
#     print(f"✅ Inserted new row for time: {next_hour_utc}")
# else:
#     print(f"⚠️ No data found for time {next_hour_utc}. Nothing inserted.")

# # --- Insert to Feature Store ---
# if not merged.empty:
#     fg.insert(merged, write_options={"wait_for_job": False})
#     print(f"✅ Inserted new row for time: {next_hour_utc}")
# else:
#     print(f"⚠️ No data found for time {next_hour_utc}. Nothing inserted.")

import os
import hopsworks
import pandas as pd
import requests
from datetime import datetime, timedelta

API_KEY = os.environ["HOPSWORKS_API_KEY"]
FG_NAME = "aqi_weather_features"
FG_VERSION = 2  # Use version 2

LAT, LON = 33.5973, 73.0479
TIMEZONE = "Asia/Karachi"

def fetch_aqi_data(start_date, end_date):
    print(f"🔍 Fetching AQI data from API for dates: {start_date} to {end_date}")
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={LAT}&longitude={LON}&hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,"
        f"nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi&start_date={start_date}&end_date={end_date}&timezone={TIMEZONE}"
    )
    response = requests.get(url).json()
    if 'hourly' not in response:
        print("⚠️ Warning: 'hourly' key missing in API response!")
        print("API response:", response)
        return pd.DataFrame()  # return empty df to avoid errors
    df = pd.DataFrame(response['hourly'])
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(TIMEZONE)
    print(f"📅 API returned {len(df)} rows with times from {df['time'].min()} to {df['time'].max()}")
    return df

def main():
    project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    # Read existing data timestamps in PKT timezone
    fg_df = fg.read()
    fg_df["time"] = pd.to_datetime(fg_df["time"], utc=True).dt.tz_convert(TIMEZONE)

    if fg_df.empty:
        print("ℹ️ Feature group empty, fetching yesterday's data as start")
        start_date = (datetime.now() - timedelta(days=1)).date().strftime("%Y-%m-%d")
    else:
        latest_time = fg_df["time"].max()
        print(f"ℹ️ Latest timestamp in feature store: {latest_time}")
        start_date = (latest_time + timedelta(hours=1)).date().strftime("%Y-%m-%d")

    end_date = datetime.now().date().strftime("%Y-%m-%d")
    print(f"ℹ️ Using start_date = {start_date}, end_date = {end_date}")

    # Fetch new data from API
    new_data = fetch_aqi_data(start_date, end_date)

    # Filter out any already existing timestamps
    existing_times = set(fg_df["time"])
    new_data = new_data[~new_data["time"].isin(existing_times)]

    if not new_data.empty:
        # Convert time to UTC before inserting
        new_data["time"] = new_data["time"].dt.tz_convert("UTC")
        fg.insert(new_data, write_options={"wait_for_job": False})
        print(f"✅ Inserted {len(new_data)} new rows from {start_date} to {end_date}")
    else:
        print("⚠️ No new data to insert")

if __name__ == "__main__":
    main()
