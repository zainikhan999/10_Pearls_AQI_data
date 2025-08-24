"""
LightGBM AQI Inference Pipeline
===============================
This script creates predictions for future AQI values using the trained LightGBM model.
It fetches the latest data, creates features, and generates 74-hour forecasts.

Author: Your Name
Date: 2025
"""

import os
import json
import requests
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import warnings
from datetime import datetime, timedelta
from typing import Tuple, List, Optional, Dict, Any

import hopsworks
from hsml.model import ModelSchema, Schema

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Hopsworks feature store configuration
FEATURE_GROUP_NAME = "aqi_weather_features"
FEATURE_GROUP_VER = 2
PREDICTIONS_FG_NAME = "aqi_predictions"
PREDICTIONS_FG_VER = 1
API_KEY = os.environ["HOPSWORKS_API_KEY"]

# Location configuration (Rawalpindi, Pakistan)
LATITUDE = 33.5973
LONGITUDE = 73.0479
TZ = "Asia/Karachi"

# Model configuration
HORIZON_H = 74  # Forecast horizon in hours (updated to match dashboard)
MAX_LAG_H = 120  # Maximum lag for features

# Directory structure
ARTIFACT_DIR = "lgb_aqi_artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "lgb_model.pkl")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "lgb_features.pkl")
TIMESTAMP_PATH = os.path.join(ARTIFACT_DIR, "last_trained_timestamp.pkl")

# Weather API configuration (OpenWeatherMap)
WEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")  # Optional for future weather data
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/forecast"

# =============================================================================
# HELPER FUNCTIONS (Same as retraining script)
# =============================================================================

def create_lag_features(df, feat_cols, lags=None):
    """
    Create lag and rolling window features for time series data.
    
    Args:
        df (pd.DataFrame): Input dataframe with time series data
        feat_cols (list): List of feature columns to create lags for
        lags (list): List of lag periods (default: [1,2,3,6,12,24,48,72,96,120])
    
    Returns:
        pd.DataFrame: DataFrame with additional lag and rolling features
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120]
    
    output_df = df.copy()
    
    for feature in feat_cols:
        # Create lag features
        for lag in lags:
            output_df[f"{feature}_lag_{lag}"] = output_df[feature].shift(lag)
        
        # Create rolling statistics
        output_df[f"{feature}_roll_mean_24"] = output_df[feature].rolling(24, min_periods=24).mean()
        output_df[f"{feature}_roll_std_24"] = output_df[feature].rolling(24, min_periods=24).std()
        output_df[f"{feature}_roll_mean_72"] = output_df[feature].rolling(72, min_periods=72).mean()
        output_df[f"{feature}_roll_std_72"] = output_df[feature].rolling(72, min_periods=72).std()
    
    return output_df


def ensure_utc(timestamp_series):
    """
    Ensure timestamp series is in UTC timezone.
    
    Args:
        timestamp_series (pd.Series): Series with timestamp data
    
    Returns:
        pd.Series: UTC timezone-aware timestamp series
    """
    ts = pd.to_datetime(timestamp_series)
    try:
        if ts.dt.tz is None:
            return ts.dt.tz_localize("UTC")
        else:
            return ts.dt.tz_convert("UTC")
    except AttributeError:
        ts = pd.to_datetime(ts, errors="coerce")
        ts = ts.dt.tz_localize("UTC")
        return ts


def utc_to_tz(timestamp_series, target_tz):
    """
    Convert UTC timestamps to specified timezone.
    
    Args:
        timestamp_series (pd.Series): UTC timestamp series
        target_tz (str): Target timezone string
    
    Returns:
        pd.Series: Converted timestamp series
    """
    utc_series = ensure_utc(timestamp_series)
    return utc_series.dt.tz_convert(target_tz)


def load_model_artifacts():
    """
    Load trained model and associated artifacts.
    
    Returns:
        tuple: (model, features, last_timestamp, model_info)
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")
    
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    
    print(f"[INFO] Loading feature list from: {FEATURES_PATH}")
    features = joblib.load(FEATURES_PATH)
    
    last_timestamp = None
    if os.path.exists(TIMESTAMP_PATH):
        last_timestamp = joblib.load(TIMESTAMP_PATH)
        print(f"[INFO] Last training timestamp: {last_timestamp}")
    
    # Load model version info if available
    model_info = {}
    version_path = os.path.join(ARTIFACT_DIR, "model_version_info.json")
    if os.path.exists(version_path):
        with open(version_path, 'r') as f:
            model_info = json.load(f)
        print(f"[INFO] Model version: {model_info.get('version', 'unknown')}")
    
    return model, features, last_timestamp, model_info


def fetch_latest_data(fg, hours_back=168):  # 7 days of historical data
    """
    Fetch the latest data from feature store for creating predictions.
    
    Args:
        fg: Feature group object
        hours_back (int): Number of hours to look back for historical data
    
    Returns:
        pd.DataFrame: Latest data sorted by time
    """
    print(f"[INFO] Fetching latest {hours_back} hours of data...")
    
    # Calculate cutoff time
    cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
    cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        # Try to fetch data with time filter
        df_raw = fg.read()
        df_raw = df_raw.sort_values("time", ascending=True).reset_index(drop=True)
        
        # Convert time column to UTC for filtering
        df_raw["time_utc"] = ensure_utc(df_raw["time"])
        
        # Filter for recent data
        recent_data = df_raw[df_raw["time_utc"] >= pd.Timestamp(cutoff_time, tz='UTC')]
        
        if len(recent_data) == 0:
            print(f"[WARNING] No recent data found. Using all available data.")
            recent_data = df_raw.tail(hours_back)  # Fallback to last N records
        
        print(f"[INFO] Fetched {len(recent_data)} recent records")
        print(f"[INFO] Data range: {recent_data['time_utc'].min()} to {recent_data['time_utc'].max()}")
        
        return recent_data
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {e}")
        raise


def create_future_timestamps(last_timestamp, horizon_hours=74):
    """
    Create future timestamps for prediction.
    
    Args:
        last_timestamp: Last available data timestamp
        horizon_hours (int): Number of hours to predict into the future
    
    Returns:
        pd.DatetimeIndex: Future timestamps in UTC
    """
    if isinstance(last_timestamp, str):
        last_timestamp = pd.Timestamp(last_timestamp)
    
    # Ensure timestamp is UTC
    if last_timestamp.tz is None:
        last_timestamp = last_timestamp.tz_localize('UTC')
    elif last_timestamp.tz != pd.Timestamp.now().tz:
        last_timestamp = last_timestamp.tz_convert('UTC')
    
    # Create hourly timestamps for the forecast horizon
    future_timestamps = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq='H',
        tz='UTC'
    )
    
    return future_timestamps


def extrapolate_features(df, future_timestamps, base_features):
    """
    Extrapolate base features for future timestamps using simple strategies.
    This is a simplified approach - in production, you might want to use
    weather forecasting APIs or more sophisticated methods.
    
    Args:
        df (pd.DataFrame): Historical data with features
        future_timestamps (pd.DatetimeIndex): Future timestamps to predict
        base_features (list): List of base feature column names
    
    Returns:
        pd.DataFrame: DataFrame with extrapolated features for future timestamps
    """
    print(f"[INFO] Extrapolating features for {len(future_timestamps)} future timestamps...")
    
    # Get the last 72 hours of data for extrapolation
    recent_data = df.tail(72)[base_features].copy()
    
    future_data = []
    
    for timestamp in future_timestamps:
        # Simple extrapolation strategies:
        # 1. Use rolling mean of last 24 hours
        # 2. Add seasonal/daily patterns
        # 3. Add some random variation to avoid static predictions
        
        row_data = {'time_utc': timestamp}
        
        for feature in base_features:
            if feature in recent_data.columns and not recent_data[feature].isna().all():
                # Strategy 1: Rolling mean with slight trend
                recent_mean = recent_data[feature].tail(24).mean()
                recent_trend = recent_data[feature].tail(12).mean() - recent_data[feature].tail(24).head(12).mean()
                
                # Strategy 2: Add daily seasonality (simple sine wave)
                hour_of_day = timestamp.hour
                daily_factor = 1 + 0.1 * np.sin(2 * np.pi * hour_of_day / 24)
                
                # Strategy 3: Add small random variation (±5%)
                noise_factor = 1 + np.random.normal(0, 0.05)
                
                # Combine strategies
                extrapolated_value = (recent_mean + recent_trend * 0.1) * daily_factor * noise_factor
                
                # Ensure non-negative values for pollutant concentrations
                extrapolated_value = max(0, extrapolated_value)
                
                row_data[feature] = extrapolated_value
            else:
                # Fallback: use the last available value or median
                if not recent_data[feature].isna().all():
                    row_data[feature] = recent_data[feature].dropna().iloc[-1]
                else:
                    row_data[feature] = df[feature].median()  # Global median as last resort
        
        future_data.append(row_data)
    
    future_df = pd.DataFrame(future_data)
    future_df = future_df.set_index('time_utc')
    
    print(f"[INFO] Feature extrapolation completed")
    return future_df


def prepare_prediction_data(df, model_features, base_features):
    """
    Prepare data for prediction by creating all necessary features.
    
    Args:
        df (pd.DataFrame): Input dataframe with historical + future data
        model_features (list): Required features for the model
        base_features (list): Base feature column names
    
    Returns:
        pd.DataFrame: Prepared data ready for prediction
    """
    print(f"[INFO] Preparing prediction data...")
    
    # Create lag features for the combined dataset
    work_df = create_lag_features(df, base_features)
    
    # Filter to only model features
    available_features = [f for f in model_features if f in work_df.columns]
    missing_features = [f for f in model_features if f not in work_df.columns]
    
    if missing_features:
        print(f"[WARNING] Missing {len(missing_features)} features: {missing_features[:5]}...")
        # Fill missing features with 0 or median values
        for feature in missing_features:
            if any(base_feat in feature for base_feat in base_features):
                # This is likely a lag/rolling feature we can't compute due to insufficient history
                work_df[feature] = 0  # or some default value
            else:
                work_df[feature] = 0
    
    # Select only the features the model expects
    prediction_data = work_df[model_features].copy()
    
    print(f"[INFO] Prediction data shape: {prediction_data.shape}")
    print(f"[INFO] Available features: {len(available_features)}/{len(model_features)}")
    
    return prediction_data


def make_predictions(model, prediction_data, future_timestamps):
    """
    Make AQI predictions using the trained model.
    
    Args:
        model: Trained LightGBM model
        prediction_data (pd.DataFrame): Prepared feature data
        future_timestamps (pd.DatetimeIndex): Future timestamps
    
    Returns:
        pd.DataFrame: Predictions with timestamps
    """
    print(f"[INFO] Making predictions for {len(future_timestamps)} timestamps...")
    
    # Get prediction data for future timestamps only
    future_data = prediction_data.loc[future_timestamps]
    
    # Handle any remaining NaN values
    future_data = future_data.fillna(0)
    
    # Make predictions
    predictions = model.predict(future_data)
    
    # Ensure predictions are reasonable (non-negative, within AQI bounds)
    predictions = np.clip(predictions, 0, 500)
    
    # Create results dataframe
    results = pd.DataFrame({
        'datetime_utc': future_timestamps,
        'datetime': utc_to_tz(pd.Series(future_timestamps), TZ),
        'predicted_us_aqi': predictions.round().astype(int),
        'prediction_date': datetime.now(tz=pd.Timestamp.now().tz)
    })
    
    print(f"[INFO] Predictions completed. AQI range: {predictions.min():.1f} - {predictions.max():.1f}")
    
    return results


def save_predictions_to_feature_store(predictions_df, fs, model_info):
    """
    Save predictions to Hopsworks feature store.
    
    Args:
        predictions_df (pd.DataFrame): Predictions dataframe
        fs: Feature store object
        model_info (dict): Model version information
    """
    print(f"[INFO] Saving {len(predictions_df)} predictions to feature store...")
    
    # Add model metadata
    predictions_df['model_version'] = model_info.get('version', 'unknown')
    predictions_df['model_name'] = model_info.get('model_name', 'lgb_aqi_forecaster')
    
    # Ensure proper column types
    predictions_df['predicted_us_aqi'] = predictions_df['predicted_us_aqi'].astype(int)
    predictions_df['model_version'] = predictions_df['model_version'].astype(str)
    
    try:
        # Get or create predictions feature group
        try:
            predictions_fg = fs.get_feature_group(name=PREDICTIONS_FG_NAME, version=PREDICTIONS_FG_VER)
            print(f"[INFO] Using existing predictions feature group")
        except:
            print(f"[INFO] Creating new predictions feature group")
            predictions_fg = fs.create_feature_group(
                name=PREDICTIONS_FG_NAME,
                version=PREDICTIONS_FG_VER,
                description="AQI predictions from LightGBM model",
                primary_key=["datetime_utc"],
                event_time="prediction_date",
                online_enabled=True
            )
        
        # Insert predictions
        predictions_fg.insert(predictions_df, write_options={"wait_for_job": True})
        print(f"[INFO] Successfully saved predictions to feature store")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save predictions to feature store: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# MAIN INFERENCE FUNCTION
# =============================================================================

def run_inference_pipeline():
    """
    Main function to run the AQI inference pipeline.
    """
    print("=" * 80)
    print("🔮 LIGHTGBM AQI INFERENCE PIPELINE")
    print("=" * 80)
    
    # Step 1: Load model artifacts
    print("\n[1/8] 📦 Loading trained model and artifacts...")
    try:
        model, model_features, last_timestamp, model_info = load_model_artifacts()
        print(f"✅ Model loaded successfully")
        print(f"📊 Model expects {len(model_features)} features")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Step 2: Connect to Hopsworks
    print("\n[2/8] 🔌 Connecting to Hopsworks...")
    try:
        project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VER)
        print("✅ Successfully connected to Hopsworks")
    except Exception as e:
        print(f"❌ Failed to connect to Hopsworks: {e}")
        return False
    
    # Step 3: Fetch latest data
    print("\n[3/8] 📊 Fetching latest data from feature store...")
    try:
        df_latest = fetch_latest_data(fg, hours_back=168)  # 7 days
        
        # Validate required columns
        base_features = ["pm_10", "pm_25", "carbon_monoxidegm", 
                        "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
        required_cols = ["time"] + base_features
        missing_cols = [col for col in required_cols if col not in df_latest.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
            
        print("✅ Latest data fetched successfully")
        
    except Exception as e:
        print(f"❌ Failed to fetch latest data: {e}")
        return False
    
    # Step 4: Create future timestamps
    print("\n[4/8] 📅 Creating future timestamps...")
    try:
        last_data_time = df_latest["time_utc"].max()
        future_timestamps = create_future_timestamps(last_data_time, HORIZON_H)
        
        print(f"✅ Created {len(future_timestamps)} future timestamps")
        print(f"📈 Forecast range: {future_timestamps[0]} to {future_timestamps[-1]}")
        
    except Exception as e:
        print(f"❌ Failed to create future timestamps: {e}")
        return False
    
    # Step 5: Extrapolate features for future timestamps
    print("\n[5/8] 🔮 Extrapolating features for future predictions...")
    try:
        # Prepare historical data
        historical_df = df_latest.set_index("time_utc")[base_features].copy()
        
        # Extrapolate features for future
        future_df = extrapolate_features(historical_df, future_timestamps, base_features)
        
        # Combine historical and future data
        combined_df = pd.concat([historical_df, future_df], axis=0).sort_index()
        
        print("✅ Feature extrapolation completed")
        
    except Exception as e:
        print(f"❌ Failed to extrapolate features: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Prepare prediction data
    print("\n[6/8] ⚙️  Preparing data for prediction...")
    try:
        prediction_data = prepare_prediction_data(combined_df, model_features, base_features)
        
        # Check if we have valid data for future timestamps
        future_pred_data = prediction_data.loc[future_timestamps]
        if future_pred_data.isna().all(axis=1).any():
            print("[WARNING] Some future timestamps have all NaN features")
        
        print("✅ Prediction data prepared")
        
    except Exception as e:
        print(f"❌ Failed to prepare prediction data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 7: Make predictions
    print("\n[7/8] 🤖 Generating AQI predictions...")
    try:
        predictions_df = make_predictions(model, prediction_data, future_timestamps)
        
        print(f"✅ Generated {len(predictions_df)} predictions")
        print(f"📊 AQI prediction range: {predictions_df['predicted_us_aqi'].min()} - {predictions_df['predicted_us_aqi'].max()}")
        
        # Display sample predictions
        print("\n📋 Sample predictions:")
        sample_preds = predictions_df.head(10)[['datetime', 'predicted_us_aqi']].copy()
        for _, row in sample_preds.iterrows():
            print(f"   {row['datetime'].strftime('%Y-%m-%d %H:%M %Z')}: AQI {row['predicted_us_aqi']}")
        
    except Exception as e:
        print(f"❌ Failed to make predictions: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 8: Save predictions to feature store
    print("\n[8/8] 💾 Saving predictions to feature store...")
    try:
        success = save_predictions_to_feature_store(predictions_df, fs, model_info)
        
        if success:
            print("✅ Predictions saved successfully to feature store")
        else:
            print("⚠️  Failed to save to feature store, but predictions generated")
        
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")
        success = False
    
    # Step 9: Save local backup
    print("\n[9/8] 💾 Saving local backup...")
    try:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(ARTIFACT_DIR, f"predictions_{timestamp_str}.csv")
        predictions_df.to_csv(backup_path, index=False)
        print(f"✅ Predictions saved to: {backup_path}")
    except Exception as e:
        print(f"⚠️  Failed to save local backup: {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 INFERENCE PIPELINE COMPLETED!")
    print("=" * 80)
    print(f"🔮 Generated predictions: {len(predictions_df)}")
    print(f"📅 Forecast horizon: {HORIZON_H} hours")
    print(f"📊 AQI range: {predictions_df['predicted_us_aqi'].min()} - {predictions_df['predicted_us_aqi'].max()}")
    print(f"⏰ Forecast starts: {future_timestamps[0].strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"⏰ Forecast ends: {future_timestamps[-1].strftime('%Y-%m-%d %H:%M UTC')}")
    
    if 'model_version' in model_info:
        print(f"🤖 Model version: {model_info['model_version']}")
    
    print("=" * 80)
    
    return True


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Run the inference pipeline.
    
    Usage:
        python aqi_inference_pipeline.py
    """
    try:
        success = run_inference_pipeline()
        if success:
            print("\n✅ Pipeline completed successfully!")
        else:
            print("\n❌ Pipeline completed with errors!")
            exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        print("\n👋 Pipeline execution finished")