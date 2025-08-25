# First, ensure you have the required libraries installed:
# !pip install hopsworks[python] lightgbm requests pandas numpy scikit-learn

# === inference_pipeline_lgbm_no_plot.py ===
import hopsworks
import joblib
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------
# Config
# ------------------
# Define constants for location and time
LATITUDE  = 33.5973
LONGITUDE = 73.0479
TZ        = "Asia/Karachi"
HORIZON_H = 72
MAX_LAG_H = 120

MODEL_NAME = "lgb_aqi_forecaster"
MODEL_VER  = 1
FEATURE_GROUP_NAME = "aqi_forecast_metrics_fg"
FEATURE_GROUP_VERSION = 1

# ------------------
# Helpers
# ------------------
def create_lag_features(df: pd.DataFrame, feat_cols, lags=None):
    """
    Creates lag and rolling mean/std features for a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        feat_cols (list): The list of columns to create features for.
        lags (list, optional): The number of hours to lag. Defaults to [1, 2, 3, 6, 12, 24, 48, 72, 96, 120].

    Returns:
        pd.DataFrame: The DataFrame with new lag features added.
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120]
    out = df.copy()
    for f in feat_cols:
        for lag in lags:
            out[f"{f}_lag_{lag}"] = out[f].shift(lag)
        out[f"{f}_roll_mean_24"] = out[f].rolling(24, min_periods=24).mean()
        out[f"{f}_roll_std_24"]  = out[f].rolling(24, min_periods=24).std()
        out[f"{f}_roll_mean_72"] = out[f].rolling(72, min_periods=72).mean()
        out[f"{f}_roll_std_72"]  = out[f].rolling(72, min_periods=72).std()
    return out

def utc_to_tz(ts_series: pd.Series, tz: str) -> pd.Series:
    """
    Converts a UTC timestamp series to a specific timezone.

    Args:
        ts_series (pd.Series): The series of UTC timestamps.
        tz (str): The target timezone string (e.g., "Asia/Karachi").

    Returns:
        pd.Series: The converted timestamp series.
    """
    s = pd.to_datetime(ts_series, utc=True)
    return s.dt.tz_convert(tz)

# ------------------
# [1] Load LightGBM model
# ------------------
print("[1] Logging into Hopsworks and loading LightGBM model...")
try:
    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    model_meta = mr.get_model(MODEL_NAME, version=None)
    model_dir = model_meta.download()

    model = joblib.load(os.path.join(model_dir, "lgb_model.pkl"))
    all_features = joblib.load(os.path.join(model_dir, "lgb_features.pkl"))
    print(f"[info] Loaded LightGBM model with {len(all_features)} features.")
except Exception as e:
    print(f"[error] Failed to load model from Hopsworks: {e}")
    # Exit the script if model loading fails
    exit()

# ------------------
# [2] Fetch pollutant forecasts
# ------------------
print("[2] Fetching pollutant forecasts from Open-Meteo...")
now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
start_utc = (now_utc + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
end_utc   = (now_utc + timedelta(hours=HORIZON_H)).strftime("%Y-%m-%dT%H:%M:%SZ")

air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
air_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
    "start": start_utc,
    "end": end_utc,
    "timezone": "UTC",
}

try:
    resp = requests.get(air_url, params=air_params)
    resp.raise_for_status()
    raw = resp.json()

    df_future = pd.DataFrame({
        "time_utc": pd.to_datetime(raw["hourly"]["time"]),
        "pm_10": raw["hourly"]["pm10"],
        "pm_25": raw["hourly"]["pm2_5"],
        "carbon_monoxidegm": raw["hourly"]["carbon_monoxide"],
        "nitrogen_dioxide": raw["hourly"]["nitrogen_dioxide"],
        "sulphur_dioxide": raw["hourly"]["sulphur_dioxide"],
        "ozone": raw["hourly"]["ozone"],
    })
    print(f"[info] Fetched {len(df_future)} future hours of pollutant data")
except requests.exceptions.RequestException as e:
    print(f"[error] Failed to fetch future data from Open-Meteo: {e}")
    exit()

# ------------------
# [3] Fetch history for lag features
# ------------------
print("[3] Fetching last 120h history...")
hist_start = (now_utc - timedelta(hours=MAX_LAG_H)).strftime("%Y-%m-%dT%H:%M:%SZ")
hist_end   = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

hist_params = air_params.copy()
hist_params["start"] = hist_start
hist_params["end"]   = hist_end

try:
    hist_resp = requests.get(air_url, params=hist_params)
    hist_resp.raise_for_status()
    hist_raw = hist_resp.json()

    df_hist = pd.DataFrame({
        "time_utc": pd.to_datetime(hist_raw["hourly"]["time"]),
        "pm_10": hist_raw["hourly"]["pm10"],
        "pm_25": hist_raw["hourly"]["pm2_5"],
        "carbon_monoxidegm": hist_raw["hourly"]["carbon_monoxide"],
        "nitrogen_dioxide": hist_raw["hourly"]["nitrogen_dioxide"],
        "sulphur_dioxide": hist_raw["hourly"]["sulphur_dioxide"],
        "ozone": hist_raw["hourly"]["ozone"],
    })
    print(f"[info] Fetched {len(df_hist)} historical hours of pollutant data")
except requests.exceptions.RequestException as e:
    print(f"[error] Failed to fetch historical data from Open-Meteo: {e}")
    exit()

# ------------------
# [4] Build features
# ------------------
print("[4] Building lag features...")
combined = pd.concat([df_hist, df_future], ignore_index=True)
combined_with_lags = create_lag_features(combined, ["pm_10","pm_25","carbon_monoxidegm","nitrogen_dioxide","sulphur_dioxide","ozone"])
combined_clean = combined_with_lags.dropna()

hist_len = len(df_hist)
future_start_idx = hist_len
valid_future_mask = combined_clean.index >= future_start_idx
future_block = combined_clean[valid_future_mask].copy()

if len(future_block) == 0:
    print("[error] No valid future data after feature engineering. Check lag requirements vs available history.")
    exit()

for c in all_features:
    if c not in future_block.columns:
        future_block[c] = 0.0

future_block = future_block[all_features]
future_indices = future_block.index - hist_len
valid_future_times = df_future.iloc[future_indices]["time_utc"].reset_index(drop=True)

print(f"[debug] Valid future timestamps: {len(valid_future_times)}")
print(f"[debug] Future block for prediction: {future_block.shape}")

# ------------------
# [5] Predict AQI using LightGBM
# ------------------
print("[5] Running LightGBM inference...")
future_pred = model.predict(future_block)

pred_df = pd.DataFrame({
    "datetime": utc_to_tz(valid_future_times, TZ),
    "predicted_us_aqi": future_pred
})

print(f"[info] Generated {len(pred_df)} predictions")

# ------------------
# [6] Fetch forecasted AQI from API and compare
# ------------------
print("[6] Fetching forecasted us_aqi from Open-Meteo API...")
aqi_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "hourly": "us_aqi",
    "timezone": TZ,
    "start_date": pred_df["datetime"].min().strftime("%Y-%m-%d"),
    "end_date": pred_df["datetime"].max().strftime("%Y-%m-%d"),
}

try:
    resp2 = requests.get(air_url, params=aqi_params)
    resp2.raise_for_status()
    raw2 = resp2.json()

    api_df = pd.DataFrame({
        "datetime": pd.to_datetime(raw2["hourly"]["time"]).tz_localize(TZ),
        "us_aqi_forecast": raw2["hourly"]["us_aqi"]
    })

    merged = pd.merge(pred_df, api_df, on="datetime", how="inner").sort_values("datetime")

    if not merged.empty:
        # Calculate comparison metrics
        mae = mean_absolute_error(merged["us_aqi_forecast"], merged["predicted_us_aqi"])
        rmse = np.sqrt(mean_squared_error(merged["us_aqi_forecast"], merged["predicted_us_aqi"]))
        corr = np.corrcoef(merged["us_aqi_forecast"].values, merged["predicted_us_aqi"].values)[0,1]
        
        print("\n✅ Comparison Metrics:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  Corr: {corr:.4f}")
        print(f"  Data points: {len(merged)}")
    else:
        print("[warn] No overlapping timestamps between model predictions and API us_aqi.")
    
    # Store predictions in Hopsworks Feature Store
    print(f"\n[7] Storing predictions in Feature Store as '{FEATURE_GROUP_NAME}'...")
    try:
        fg = fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            description="Predicted and API-provided AQI forecasts",
            primary_key=["datetime"],
            event_time="datetime",
            online_enabled=True,
        )
        fg.insert(merged)
        print("✅ Successfully wrote data to Feature Store.")
    except Exception as e:
        print(f"[error] Failed to write to Feature Store: {e}")

except requests.exceptions.RequestException as e:
    print(f"[error] Failed to fetch API us_aqi data: {e}")

