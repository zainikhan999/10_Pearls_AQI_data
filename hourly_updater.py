import os
import hopsworks
import pandas as pd
import requests
from datetime import datetime, timedelta
from pytz import timezone

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
        return pd.DataFrame()  # empty df to avoid errors
    df = pd.DataFrame(response['hourly'])

    # Rename columns to match feature store schema
    df = df.rename(columns={
        "pm10": "pm_10",
        "pm2_5": "pm_25",
        "carbon_monoxide": "carbon_monoxidegm",
    })

    # Fix data types to match schema
    df['carbon_monoxidegm'] = df['carbon_monoxidegm'].astype('Int64')  # nullable int
    df['ozone'] = df['ozone'].astype('Int64')  # nullable int
    df['carbon_dioxide'] = df['carbon_dioxide'].astype(float)
    df['pm_10'] = df['pm_10'].astype(float)
    df['pm_25'] = df['pm_25'].astype(float)

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
        # Start from yesterday date (string)
        start_dt = datetime.now(tz=pd.Timestamp.now().tz).replace(minute=0, second=0, microsecond=0) - timedelta(days=1)
    else:
        latest_time = fg_df["time"].max()
        print(f"ℹ️ Latest timestamp in feature store: {latest_time}")
        start_dt = latest_time + timedelta(hours=1)

    # Calculate end datetime as current time truncated to hour

    tz = timezone("Asia/Karachi")

    now = datetime.now(tz).replace(minute=0, second=0, microsecond=0)


    
    if start_dt > now:
        print(f"⚠️ Start datetime {start_dt} is after current hour {now}. No new data to fetch.")
        return

    start_date = start_dt.date().strftime("%Y-%m-%d")
    end_date = now.date().strftime("%Y-%m-%d")

    print(f"ℹ️ Using start_date = {start_date}, end_date = {end_date}")

    # Fetch data from API
    new_data = fetch_aqi_data(start_date, end_date)

    if new_data.empty:
        print("⚠️ No data returned from API")
        return

    # Filter out times <= latest_time (exact datetime comparison)
    if not fg_df.empty:
        new_data = new_data[new_data["time"] >= start_dt]

    # Only keep data up to current hour (inclusive)
    new_data = new_data[new_data["time"] <= now]

    if not new_data.empty:
        # Convert time to UTC before inserting
        new_data["time"] = new_data["time"].dt.tz_convert("UTC")
        fg.insert(new_data, write_options={"wait_for_job": False})
        print(f"✅ Inserted {len(new_data)} new rows from {start_date} to {end_date} (up to {now})")
    else:
        print("⚠️ No new data to insert after filtering existing timestamps and future hours")

if __name__ == "__main__":
    main()
