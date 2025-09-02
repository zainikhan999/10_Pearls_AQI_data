import hopsworks
import joblib
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pytz import timezone as pytz_timezone

# ------------------
# Config
# ------------------
LATITUDE  = 33.5973
LONGITUDE = 73.0479
TZ        = "Asia/Karachi"
HORIZON_H = 72
MAX_LAG_H = 120

MODEL_NAME = "lgb_aqi_forecaster"
MODEL_VER  = 2  # Use latest version
FEATURE_GROUP_NAME = "aqi_forecast_metrics_fg"
FEATURE_GROUP_VERSION = 1

# ------------------
# Fixed Helper Functions
# ------------------
def create_lag_features(df: pd.DataFrame, feat_cols, lags=None):
    """Creates lag and rolling mean/std features for a given DataFrame."""
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120]
    out = df.copy()
    for f in feat_cols:
        for lag in lags:
            out[f"{f}_lag_{lag}"] = out[f].shift(lag)
        out[f"{f}_roll_mean_24"] = out[f].rolling(24, min_periods=12).mean()  # Reduced min_periods
        out[f"{f}_roll_std_24"]  = out[f].rolling(24, min_periods=12).std()
        out[f"{f}_roll_mean_72"] = out[f].rolling(72, min_periods=24).mean()  # Reduced min_periods
        out[f"{f}_roll_std_72"]  = out[f].rolling(72, min_periods=24).std()
    return out

def utc_to_tz(ts_series: pd.Series, tz: str) -> pd.Series:
    """Converts a UTC timestamp series to a specific timezone."""
    s = pd.to_datetime(ts_series, utc=True)
    return s.dt.tz_convert(tz)

def fetch_historical_and_forecast_data():
    """Fetch both historical data for lag features and forecast data."""
    print("🔍 Fetching comprehensive data from API...")
    
    # Get current time in PKT
    pkt_tz = pytz_timezone(TZ)
    now_pkt = datetime.now(pkt_tz).replace(minute=0, second=0, microsecond=0)
    now_utc = now_pkt.astimezone(timezone.utc)
    
    # Fetch historical data for lag features (120 hours back)
    hist_start = (now_utc - timedelta(hours=MAX_LAG_H)).strftime("%Y-%m-%d")
    hist_end = now_utc.strftime("%Y-%m-%d")
    
    # Fetch forecast data (up to 72 hours ahead)
    forecast_end = (now_utc + timedelta(hours=HORIZON_H)).strftime("%Y-%m-%d")
    
    print(f"Historical period: {hist_start} to {hist_end}")
    print(f"Forecast period: {hist_end} to {forecast_end}")
    
    # Single API call for complete data range
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "start_date": hist_start,
        "end_date": forecast_end,
        "timezone": TZ,  # Get data directly in PKT timezone
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'hourly' not in data:
            print("⚠️ Warning: 'hourly' key missing in API response!")
            return pd.DataFrame(), pd.DataFrame()
            
        df = pd.DataFrame({
            "time": pd.to_datetime(data["hourly"]["time"]).dt.tz_localize(TZ),
            "pm_10": data["hourly"]["pm10"],
            "pm_25": data["hourly"]["pm2_5"],
            "carbon_monoxidegm": data["hourly"]["carbon_monoxide"],
            "nitrogen_dioxide": data["hourly"]["nitrogen_dioxide"],
            "sulphur_dioxide": data["hourly"]["sulphur_dioxide"],
            "ozone": data["hourly"]["ozone"],
            "us_aqi": data["hourly"]["us_aqi"]
        })
        
        print(f"📅 API returned {len(df)} total rows")
        print(f"Time range: {df['time'].min()} to {df['time'].max()}")
        
        # Split into historical and forecast parts
        hist_df = df[df['time'] <= now_pkt].copy()
        forecast_df = df[df['time'] > now_pkt].copy()
        
        print(f"Historical data: {len(hist_df)} rows")
        print(f"Forecast data: {len(forecast_df)} rows")
        
        return hist_df, forecast_df
        
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def fill_missing_dates(df, start_time, end_time):
    """Fill missing hourly timestamps with NaN values."""
    print("🔧 Filling missing timestamps...")
    
    # Create complete hourly range
    complete_range = pd.date_range(start=start_time, end=end_time, freq='H', tz=TZ)
    complete_df = pd.DataFrame({'time': complete_range})
    
    # Merge with existing data
    filled_df = pd.merge(complete_df, df, on='time', how='left')
    
    print(f"Original: {len(df)} rows, After filling: {len(filled_df)} rows")
    return filled_df

def main():
    print("🚀 Starting Enhanced AQI Prediction Pipeline...")
    
    # Connect to Hopsworks
    try:
        project = hopsworks.login()
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        
        # Load model with error handling for version
        try:
            model_meta = mr.get_model(MODEL_NAME, version=MODEL_VER)
        except:
            print(f"Model version {MODEL_VER} not found, trying latest version...")
            models = mr.get_models(MODEL_NAME)
            model_meta = max(models, key=lambda x: x.version)
            print(f"Using model version {model_meta.version}")
            
        model_dir = model_meta.download()
        model = joblib.load(os.path.join(model_dir, "lgb_model.pkl"))
        all_features = joblib.load(os.path.join(model_dir, "lgb_features.pkl"))
        print(f"✅ Loaded model with {len(all_features)} features")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Fetch data
    hist_df, forecast_df = fetch_historical_and_forecast_data()
    if hist_df.empty or forecast_df.empty:
        print("❌ Failed to fetch required data")
        return

    # Get current time in PKT
    pkt_tz = pytz_timezone(TZ)
    now_pkt = datetime.now(pkt_tz).replace(minute=0, second=0, microsecond=0)
    
    # Fill missing timestamps in forecast period
    forecast_start = now_pkt + timedelta(hours=1)
    forecast_end = now_pkt + timedelta(hours=HORIZON_H)
    forecast_df = fill_missing_dates(forecast_df, forecast_start, forecast_end)
    
    # Combine historical and forecast data for feature engineering
    print("🔧 Building lag features...")
    combined_df = pd.concat([hist_df, forecast_df], ignore_index=True).sort_values('time')
    
    # Create lag features
    feature_cols = ["pm_10", "pm_25", "carbon_monoxidegm", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
    combined_with_features = create_lag_features(combined_df, feature_cols)
    
    # Extract forecast portion with valid features
    forecast_start_idx = len(hist_df)
    forecast_with_features = combined_with_features.iloc[forecast_start_idx:].copy()
    
    # Remove rows with NaN in critical lag features (but be more lenient)
    critical_features = [f"{feat}_lag_1" for feat in feature_cols]
    valid_forecast = forecast_with_features.dropna(subset=critical_features)
    
    if valid_forecast.empty:
        print("❌ No valid forecast data after feature engineering")
        return
    
    print(f"📊 Valid forecast period: {len(valid_forecast)} hours")
    print(f"From: {valid_forecast['time'].min()} To: {valid_forecast['time'].max()}")
    
    # Ensure all model features are present
    for feature in all_features:
        if feature not in valid_forecast.columns:
            valid_forecast[feature] = 0.0
    
    # Make predictions
    print("🎯 Generating predictions...")
    X_forecast = valid_forecast[all_features]
    predictions = model.predict(X_forecast)
    
    # Create prediction dataframe
    pred_df = pd.DataFrame({
        "datetime": valid_forecast['time'].values,
        "predicted_us_aqi": predictions
    })
    
    # Fetch API forecasts for comparison
    print("📡 Fetching API us_aqi forecasts for comparison...")
    try:
        api_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        api_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": "us_aqi",
            "start_date": pred_df["datetime"].min().strftime("%Y-%m-%d"),
            "end_date": pred_df["datetime"].max().strftime("%Y-%m-%d"),
            "timezone": TZ,
        }
        
        api_resp = requests.get(api_url, params=api_params)
        api_resp.raise_for_status()
        api_data = api_resp.json()
        
        api_df = pd.DataFrame({
            "datetime": pd.to_datetime(api_data["hourly"]["time"]).dt.tz_localize(TZ),
            "us_aqi_forecast": api_data["hourly"]["us_aqi"]
        })
        
        # Merge predictions with API forecasts
        final_df = pd.merge(pred_df, api_df, on="datetime", how="left")
        
        # Calculate metrics where both values exist
        valid_comparison = final_df.dropna(subset=["predicted_us_aqi", "us_aqi_forecast"])
        if not valid_comparison.empty:
            mae = mean_absolute_error(valid_comparison["us_aqi_forecast"], valid_comparison["predicted_us_aqi"])
            rmse = np.sqrt(mean_squared_error(valid_comparison["us_aqi_forecast"], valid_comparison["predicted_us_aqi"]))
            corr = np.corrcoef(valid_comparison["us_aqi_forecast"], valid_comparison["predicted_us_aqi"])[0,1]
            
            print(f"\n📈 Comparison Metrics (n={len(valid_comparison)}):")
            print(f"  MAE:  {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  Corr: {corr:.4f}")
        
    except Exception as e:
        print(f"⚠️ Could not fetch API forecasts for comparison: {e}")
        final_df = pred_df.copy()
        final_df["us_aqi_forecast"] = None

    # Store results
    print(f"\n💾 Storing {len(final_df)} predictions in Feature Store...")
    try:
        fg = fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            description="Enhanced AQI predictions with complete temporal coverage",
            primary_key=["datetime"],
            event_time="datetime",
            online_enabled=False,
            statistics_config={}
        )
        
        # Convert timezone for storage (if needed)
        store_df = final_df.copy()
        store_df["datetime"] = store_df["datetime"].dt.tz_convert("UTC")
        
        fg.insert(store_df, write_options={"wait_for_job": False})
        print("✅ Successfully stored predictions")
        
        # Print summary
        print(f"\n📋 PREDICTION SUMMARY:")
        print(f"  Total predictions: {len(final_df)}")
        print(f"  Date range: {final_df['datetime'].min().strftime('%Y-%m-%d %H:%M')} to {final_df['datetime'].max().strftime('%Y-%m-%d %H:%M')}")
        print(f"  Prediction range: {final_df['predicted_us_aqi'].min():.1f} - {final_df['predicted_us_aqi'].max():.1f}")
        
        # Show daily breakdown
        daily_counts = final_df.groupby(final_df['datetime'].dt.date).size()
        print(f"\n📅 Daily prediction counts:")
        for date, count in daily_counts.items():
            print(f"  {date}: {count} predictions")
            
    except Exception as e:
        print(f"❌ Failed to store predictions: {e}")

if __name__ == "__main__":
    main()