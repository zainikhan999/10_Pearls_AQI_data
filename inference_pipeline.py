"""
LightGBM AQI Inference Pipeline
===============================
This script creates predictions for future AQI values using the trained LightGBM model.
It loads the latest model from Hopsworks Model Registry and generates 74-hour forecasts.

Author: Your Name
Date: 2025
"""

import os
import json
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

# Hopsworks configuration
FEATURE_GROUP_NAME = "aqi_weather_features"
FEATURE_GROUP_VER = 2
PREDICTIONS_FG_NAME = "aqi_predictions"
PREDICTIONS_FG_VER = 1
API_KEY = os.environ["HOPSWORKS_API_KEY"]

# Model configuration
MODEL_NAME = "lgb_aqi_forecaster"
HORIZON_H = 74  # Forecast horizon in hours
TZ = "Asia/Karachi"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_lag_features(df, feat_cols, lags=None):
    """Create lag and rolling window features for time series data."""
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
    """Ensure timestamp series is in UTC timezone."""
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
    """Convert UTC timestamps to specified timezone."""
    utc_series = ensure_utc(timestamp_series)
    return utc_series.dt.tz_convert(target_tz)


def load_latest_model_from_registry(mr):
    """
    Load the latest model version from Hopsworks Model Registry.
    
    Args:
        mr: Model registry object
    
    Returns:
        tuple: (model, features, model_info)
    """
    try:
        print(f"Loading latest model '{MODEL_NAME}' from Model Registry...")
        
        # Get all versions of the model
        models = mr.get_models(MODEL_NAME)
        if not models:
            raise ValueError(f"No models found with name '{MODEL_NAME}'")
        
        # Get the latest version
        latest_model = max(models, key=lambda x: x.version)
        model_version = latest_model.version
        print(f"Latest model version: {model_version}")
        
        # Download model artifacts
        model_dir = latest_model.download()
        print(f"Model downloaded to: {model_dir}")
        
        # Load model and features
        model_file = os.path.join(model_dir, "lgb_model.pkl")
        features_file = os.path.join(model_dir, "lgb_features.pkl")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        model = joblib.load(model_file)
        features = joblib.load(features_file)
        
        # Extract model info
        model_info = {
            'model_name': MODEL_NAME,
            'version': model_version,
            'description': getattr(latest_model, 'description', ''),
            'training_metrics': getattr(latest_model, 'training_metrics', {})
        }
        
        print(f"Successfully loaded model version {model_version} with {len(features)} features")
        
        return model, features, model_info
        
    except Exception as e:
        print(f"Failed to load model from registry: {e}")
        raise


def fetch_latest_data(fg, hours_back=168):
    """Fetch the latest data from feature store."""
    print(f"Fetching latest {hours_back} hours of data...")
    
    try:
        df_raw = fg.read()
        df_raw = df_raw.sort_values("time", ascending=True).reset_index(drop=True)
        
        # Convert time to UTC
        df_raw["time_utc"] = ensure_utc(df_raw["time"])
        
        # Get recent data
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        recent_data = df_raw[df_raw["time_utc"] >= pd.Timestamp(cutoff_time, tz='UTC')]
        
        if len(recent_data) == 0:
            print("No recent data found. Using all available data.")
            recent_data = df_raw.tail(hours_back)
        
        print(f"Fetched {len(recent_data)} records")
        print(f"Data range: {recent_data['time_utc'].min()} to {recent_data['time_utc'].max()}")
        
        return recent_data
        
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        raise


def create_future_timestamps(last_timestamp, horizon_hours=74):
    """Create future timestamps for prediction."""
    if isinstance(last_timestamp, str):
        last_timestamp = pd.Timestamp(last_timestamp)
    
    if last_timestamp.tz is None:
        last_timestamp = last_timestamp.tz_localize('UTC')
    elif last_timestamp.tz != pd.Timestamp.now().tz:
        last_timestamp = last_timestamp.tz_convert('UTC')
    
    future_timestamps = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq='H',
        tz='UTC'
    )
    
    return future_timestamps


def extrapolate_features(df, future_timestamps, base_features):
    """Extrapolate base features for future timestamps."""
    print(f"Extrapolating features for {len(future_timestamps)} future timestamps...")
    
    recent_data = df.tail(72)[base_features].copy()
    future_data = []
    
    for timestamp in future_timestamps:
        row_data = {'time_utc': timestamp}
        
        for feature in base_features:
            if feature in recent_data.columns and not recent_data[feature].isna().all():
                # Use rolling mean with slight trend and daily seasonality
                recent_mean = recent_data[feature].tail(24).mean()
                recent_trend = recent_data[feature].tail(12).mean() - recent_data[feature].tail(24).head(12).mean()
                
                # Add daily seasonality
                hour_of_day = timestamp.hour
                daily_factor = 1 + 0.1 * np.sin(2 * np.pi * hour_of_day / 24)
                
                # Add small random variation
                noise_factor = 1 + np.random.normal(0, 0.05)
                
                extrapolated_value = (recent_mean + recent_trend * 0.1) * daily_factor * noise_factor
                extrapolated_value = max(0, extrapolated_value)  # Ensure non-negative
                
                row_data[feature] = extrapolated_value
            else:
                # Fallback to last available value or median
                if not recent_data[feature].isna().all():
                    row_data[feature] = recent_data[feature].dropna().iloc[-1]
                else:
                    row_data[feature] = df[feature].median()
        
        future_data.append(row_data)
    
    future_df = pd.DataFrame(future_data).set_index('time_utc')
    print("Feature extrapolation completed")
    return future_df


def prepare_prediction_data(df, model_features, base_features):
    """Prepare data for prediction by creating all necessary features."""
    print("Preparing prediction data...")
    
    # Create lag features
    work_df = create_lag_features(df, base_features)
    
    # Handle missing features
    missing_features = [f for f in model_features if f not in work_df.columns]
    if missing_features:
        print(f"Filling {len(missing_features)} missing features with zeros")
        for feature in missing_features:
            work_df[feature] = 0
    
    # Select only model features
    prediction_data = work_df[model_features].copy()
    print(f"Prediction data shape: {prediction_data.shape}")
    
    return prediction_data


def make_predictions(model, prediction_data, future_timestamps):
    """Make AQI predictions using the trained model."""
    print(f"Making predictions for {len(future_timestamps)} timestamps...")
    
    # Get future data and fill NaNs
    future_data = prediction_data.loc[future_timestamps].fillna(0)
    
    # Make predictions
    predictions = model.predict(future_data)
    predictions = np.clip(predictions, 0, 500)  # Ensure reasonable AQI bounds
    
    # Create results
    results = pd.DataFrame({
        'datetime_utc': future_timestamps,
        'datetime': utc_to_tz(pd.Series(future_timestamps), TZ),
        'predicted_us_aqi': predictions.round().astype(int),
        'prediction_date': datetime.now(tz=pd.Timestamp.now().tz)
    })
    
    print(f"Predictions completed. AQI range: {predictions.min():.1f} - {predictions.max():.1f}")
    return results


def save_predictions_to_feature_store(predictions_df, fs, model_info):
    """Save predictions to Hopsworks feature store."""
    print(f"Saving {len(predictions_df)} predictions to feature store...")
    
    # Add model metadata
    predictions_df['model_version'] = str(model_info.get('version', 'unknown'))
    predictions_df['model_name'] = model_info.get('model_name', MODEL_NAME)
    predictions_df['predicted_us_aqi'] = predictions_df['predicted_us_aqi'].astype(int)
    
    try:
        # Get or create predictions feature group
        try:
            predictions_fg = fs.get_feature_group(name=PREDICTIONS_FG_NAME, version=PREDICTIONS_FG_VER)
            print("Using existing predictions feature group")
        except:
            print("Creating new predictions feature group")
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
        print("Successfully saved predictions to feature store")
        return True
        
    except Exception as e:
        print(f"Failed to save predictions to feature store: {e}")
        return False


# =============================================================================
# MAIN INFERENCE FUNCTION
# =============================================================================

def run_inference_pipeline():
    """Main function to run the AQI inference pipeline."""
    print("=" * 80)
    print("LIGHTGBM AQI INFERENCE PIPELINE")
    print("=" * 80)
    
    # Step 1: Connect to Hopsworks
    print("\n[1/7] Connecting to Hopsworks...")
    try:
        project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VER)
        print("Successfully connected to Hopsworks")
    except Exception as e:
        print(f"Failed to connect to Hopsworks: {e}")
        return False
    
    # Step 2: Load latest model from registry
    print("\n[2/7] Loading latest model from Model Registry...")
    try:
        model, model_features, model_info = load_latest_model_from_registry(mr)
        print(f"Model loaded successfully")
        print(f"Model expects {len(model_features)} features")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False
    
    # Step 3: Fetch latest data
    print("\n[3/7] Fetching latest data from feature store...")
    try:
        df_latest = fetch_latest_data(fg, hours_back=168)  # 7 days
        
        base_features = ["pm_10", "pm_25", "carbon_monoxidegm", 
                        "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
        required_cols = ["time"] + base_features
        missing_cols = [col for col in required_cols if col not in df_latest.columns]
        
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
            
        print("Latest data fetched successfully")
        
    except Exception as e:
        print(f"Failed to fetch latest data: {e}")
        return False
    
    # Step 4: Create future timestamps
    print("\n[4/7] Creating future timestamps...")
    try:
        last_data_time = df_latest["time_utc"].max()
        future_timestamps = create_future_timestamps(last_data_time, HORIZON_H)
        
        print(f"Created {len(future_timestamps)} future timestamps")
        print(f"Forecast range: {future_timestamps[0]} to {future_timestamps[-1]}")
        
    except Exception as e:
        print(f"Failed to create future timestamps: {e}")
        return False
    
    # Step 5: Extrapolate features
    print("\n[5/7] Extrapolating features for future predictions...")
    try:
        historical_df = df_latest.set_index("time_utc")[base_features].copy()
        future_df = extrapolate_features(historical_df, future_timestamps, base_features)
        combined_df = pd.concat([historical_df, future_df], axis=0).sort_index()
        print("Feature extrapolation completed")
        
    except Exception as e:
        print(f"Failed to extrapolate features: {e}")
        return False
    
    # Step 6: Prepare prediction data and make predictions
    print("\n[6/7] Preparing data and generating predictions...")
    try:
        prediction_data = prepare_prediction_data(combined_df, model_features, base_features)
        predictions_df = make_predictions(model, prediction_data, future_timestamps)
        
        print(f"Generated {len(predictions_df)} predictions")
        print(f"AQI prediction range: {predictions_df['predicted_us_aqi'].min()} - {predictions_df['predicted_us_aqi'].max()}")
        
        # Display sample predictions
        print("\nSample predictions:")
        for _, row in predictions_df.head(10).iterrows():
            print(f"   {row['datetime'].strftime('%Y-%m-%d %H:%M %Z')}: AQI {row['predicted_us_aqi']}")
        
    except Exception as e:
        print(f"Failed to make predictions: {e}")
        return False
    
    # Step 7: Save predictions
    print("\n[7/7] Saving predictions to feature store...")
    try:
        success = save_predictions_to_feature_store(predictions_df, fs, model_info)
        
        # Save local backup regardless
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"predictions_{timestamp_str}.csv"
        predictions_df.to_csv(backup_path, index=False)
        print(f"Local backup saved to: {backup_path}")
        
        if success:
            print("Predictions saved successfully to feature store")
        else:
            print("Failed to save to feature store, but local backup created")
        
    except Exception as e:
        print(f"Error saving predictions: {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("INFERENCE PIPELINE COMPLETED!")
    print("=" * 80)
    print(f"Generated predictions: {len(predictions_df)}")
    print(f"Forecast horizon: {HORIZON_H} hours")
    print(f"AQI range: {predictions_df['predicted_us_aqi'].min()} - {predictions_df['predicted_us_aqi'].max()}")
    print(f"Forecast starts: {future_timestamps[0].strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Forecast ends: {future_timestamps[-1].strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Model version: {model_info.get('version', 'unknown')}")
    print("=" * 80)
    
    return True


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    """Run the inference pipeline."""
    try:
        success = run_inference_pipeline()
        if success:
            print("\nPipeline completed successfully!")
        else:
            print("\nPipeline completed with errors!")
            exit(1)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        print("\nPipeline execution finished")