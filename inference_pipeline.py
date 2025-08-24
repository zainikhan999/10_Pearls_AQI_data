"""
LightGBM AQI Inference Pipeline - Improved Accurate Version
===========================================================
This script combines the accuracy of real forecast data with robust Hopsworks integration.

Key improvements:
1. Uses real pollutant forecasts from Open-Meteo API instead of synthetic data
2. Proper historical data fetching for lag features
3. Maintains robust Hopsworks integration with fixed primary key
4. Better error handling and validation
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import requests
import warnings
from datetime import datetime, timedelta, timezone
from typing import Tuple, List, Optional, Dict, Any

import hopsworks

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Location configuration (Rawalpindi, Pakistan)
LATITUDE = 33.5973
LONGITUDE = 73.0479

# Hopsworks configuration
FEATURE_GROUP_NAME = "aqi_weather_features"
FEATURE_GROUP_VER = 2
PREDICTIONS_FG_NAME = "aqi_predictions"
PREDICTIONS_FG_VER = 1
API_KEY = os.environ["HOPSWORKS_API_KEY"]

# Model configuration
MODEL_NAME = "lgb_aqi_forecaster"
HORIZON_H = 72  # Forecast horizon in hours
MAX_LAG_H = 120  # Maximum lag hours needed for features
TZ = "Asia/Karachi"

# Base pollutant features expected by the model
BASE_FEATURES = ["pm_10", "pm_25", "carbon_monoxidegm", 
                "nitrogen_dioxide", "sulphur_dioxide", "ozone"]

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


def utc_to_tz(timestamp_series, target_tz):
    """Convert UTC timestamps to specified timezone."""
    s = pd.to_datetime(timestamp_series, utc=True)
    return s.dt.tz_convert(target_tz)


def load_latest_model_from_registry(mr):
    """Load the latest model version from Hopsworks Model Registry."""
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


def fetch_pollutant_forecasts(hours=72):
    """Fetch real pollutant forecasts from Open-Meteo API."""
    print(f"Fetching {hours}h pollutant forecasts from Open-Meteo API...")
    
    try:
        now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_utc = (now_utc + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_utc = (now_utc + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        air_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
            "start": start_utc,
            "end": end_utc,
            "timezone": "UTC",
        }
        
        response = requests.get(air_url, params=air_params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        df_forecast = pd.DataFrame({
            "time_utc": pd.to_datetime(data["hourly"]["time"]),
            "pm_10": data["hourly"]["pm10"],
            "pm_25": data["hourly"]["pm2_5"],
            "carbon_monoxidegm": data["hourly"]["carbon_monoxide"],
            "nitrogen_dioxide": data["hourly"]["nitrogen_dioxide"],
            "sulphur_dioxide": data["hourly"]["sulphur_dioxide"],
            "ozone": data["hourly"]["ozone"],
        })
        
        print(f"Successfully fetched {len(df_forecast)} hours of forecast data")
        print(f"Forecast range: {df_forecast['time_utc'].min()} to {df_forecast['time_utc'].max()}")
        
        return df_forecast
        
    except Exception as e:
        print(f"Failed to fetch pollutant forecasts: {e}")
        raise


def fetch_historical_pollutants(hours=120):
    """Fetch historical pollutant data for lag features."""
    print(f"Fetching last {hours}h historical pollutant data...")
    
    try:
        now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        hist_start = (now_utc - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        hist_end = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        hist_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
            "start": hist_start,
            "end": hist_end,
            "timezone": "UTC",
        }
        
        response = requests.get(air_url, params=hist_params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        df_historical = pd.DataFrame({
            "time_utc": pd.to_datetime(data["hourly"]["time"]),
            "pm_10": data["hourly"]["pm10"],
            "pm_25": data["hourly"]["pm2_5"],
            "carbon_monoxidegm": data["hourly"]["carbon_monoxide"],
            "nitrogen_dioxide": data["hourly"]["nitrogen_dioxide"],
            "sulphur_dioxide": data["hourly"]["sulphur_dioxide"],
            "ozone": data["hourly"]["ozone"],
        })
        
        print(f"Successfully fetched {len(df_historical)} hours of historical data")
        print(f"Historical range: {df_historical['time_utc'].min()} to {df_historical['time_utc'].max()}")
        
        return df_historical
        
    except Exception as e:
        print(f"Failed to fetch historical data: {e}")
        raise


def prepare_forecast_features(df_historical, df_forecast, model_features):
    """Prepare features for forecasting using real historical and forecast data."""
    print("Preparing forecast features with real data...")
    
    try:
        # Combine historical and forecast data
        combined_df = pd.concat([df_historical, df_forecast], ignore_index=True)
        print(f"Combined data shape: {combined_df.shape}")
        
        # Create lag features on combined dataset
        combined_with_lags = create_lag_features(combined_df, BASE_FEATURES)
        print(f"Shape after lag features: {combined_with_lags.shape}")
        
        # Remove rows with NaN values (mainly from lag feature creation)
        combined_clean = combined_with_lags.dropna()
        print(f"Shape after dropping NaN: {combined_clean.shape}")
        
        # Identify which rows correspond to our forecast period
        hist_len = len(df_historical)
        future_start_idx = hist_len
        
        # Get valid future rows that survived NaN filtering
        valid_future_mask = combined_clean.index >= future_start_idx
        future_block = combined_clean[valid_future_mask].copy()
        print(f"Valid future data shape: {future_block.shape}")
        
        if len(future_block) == 0:
            raise ValueError("No valid future data after feature engineering. Increase historical data or reduce lag requirements.")
        
        # Ensure all model features are present
        missing_features = []
        for feature in model_features:
            if feature not in future_block.columns:
                future_block[feature] = 0.0
                missing_features.append(feature)
        
        if missing_features:
            print(f"Warning: Filled {len(missing_features)} missing features with zeros")
        
        # Select only model features in correct order
        prediction_data = future_block[model_features].copy()
        
        # Get corresponding timestamps
        future_indices = future_block.index - hist_len
        valid_timestamps = df_forecast.iloc[future_indices]["time_utc"].reset_index(drop=True)
        
        print(f"Prediction data ready: {prediction_data.shape}")
        print(f"Valid timestamps: {len(valid_timestamps)}")
        
        return prediction_data, valid_timestamps
        
    except Exception as e:
        print(f"Failed to prepare forecast features: {e}")
        raise


def make_aqi_predictions(model, prediction_data, timestamps):
    """Generate AQI predictions using the trained model."""
    print(f"Making AQI predictions for {len(timestamps)} timestamps...")
    
    try:
        # Make predictions
        predictions = model.predict(prediction_data)
        predictions = np.clip(predictions, 0, 500)  # Ensure reasonable AQI bounds
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'prediction_id': [f"pred_{ts.strftime('%Y%m%d_%H%M%S')}" for ts in timestamps],
            'datetime_utc': timestamps,
            'datetime_str': [ts.strftime('%Y-%m-%d %H:%M:%S UTC') for ts in timestamps],
            'datetime': utc_to_tz(pd.Series(timestamps), TZ),
            'predicted_us_aqi': predictions.round().astype(int),
            'prediction_date': datetime.now(),
            'forecast_hour': list(range(1, len(timestamps) + 1))
        })
        
        print(f"Predictions completed!")
        print(f"AQI range: {predictions.min():.1f} - {predictions.max():.1f}")
        print(f"Mean AQI: {predictions.mean():.1f}")
        
        return results_df
        
    except Exception as e:
        print(f"Failed to make predictions: {e}")
        raise


def validate_against_api_forecast(predictions_df):
    """Validate our predictions against API's AQI forecasts."""
    print("Validating predictions against API AQI forecasts...")
    
    try:
        # Fetch AQI forecasts from API
        air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        aqi_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": "us_aqi",
            "timezone": TZ,
            "start_date": predictions_df["datetime"].min().strftime("%Y-%m-%d"),
            "end_date": predictions_df["datetime"].max().strftime("%Y-%m-%d"),
        }
        
        response = requests.get(air_url, params=aqi_params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        api_df = pd.DataFrame({
            "datetime": pd.to_datetime(data["hourly"]["time"]).tz_localize(TZ),
            "api_us_aqi": data["hourly"]["us_aqi"]
        })
        
        # Merge predictions with API forecasts
        comparison_df = pd.merge(predictions_df, api_df, on="datetime", how="inner")
        
        if len(comparison_df) > 0:
            mae = np.mean(np.abs(comparison_df["predicted_us_aqi"] - comparison_df["api_us_aqi"]))
            rmse = np.sqrt(np.mean((comparison_df["predicted_us_aqi"] - comparison_df["api_us_aqi"]) ** 2))
            corr = np.corrcoef(comparison_df["predicted_us_aqi"], comparison_df["api_us_aqi"])[0, 1]
            
            print(f"Validation metrics:")
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  Correlation: {corr:.4f}")
            print(f"  Data points: {len(comparison_df)}")
            
            # Save comparison
            comparison_df.to_csv("prediction_validation.csv", index=False)
            print("Saved validation results to: prediction_validation.csv")
            
            return {
                'mae': mae,
                'rmse': rmse,
                'correlation': corr,
                'data_points': len(comparison_df)
            }
        else:
            print("Warning: No overlapping data points for validation")
            return None
            
    except Exception as e:
        print(f"Validation failed: {e}")
        return None


def save_predictions_to_feature_store(predictions_df, fs, model_info):
    """Save predictions to Hopsworks feature store."""
    print(f"Saving {len(predictions_df)} predictions to feature store...")
    
    # Create a copy and prepare for Hopsworks
    predictions_copy = predictions_df.copy()
    
    # Add model metadata
    predictions_copy['model_version'] = str(model_info.get('version', 'unknown'))
    predictions_copy['model_name'] = model_info.get('model_name', MODEL_NAME)
    
    # Convert datetime columns to timezone-naive
    for col in ['datetime_utc', 'prediction_date', 'datetime']:
        if col in predictions_copy.columns:
            predictions_copy[col] = pd.to_datetime(predictions_copy[col]).dt.tz_localize(None)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to save to feature store...")
            
            # Try to get existing feature group
            try:
                predictions_fg = fs.get_feature_group(name=PREDICTIONS_FG_NAME, version=PREDICTIONS_FG_VER)
                print(f"Found existing predictions feature group")
            except:
                print("Creating new predictions feature group...")
                predictions_fg = fs.create_feature_group(
                    name=PREDICTIONS_FG_NAME,
                    version=PREDICTIONS_FG_VER,
                    description="AQI predictions from LightGBM model using real forecast data",
                    primary_key=["prediction_id"],
                    event_time="prediction_date",
                    online_enabled=False,
                    statistics_config=False
                )
                print("Feature group created successfully")
            
            # Insert data
            predictions_fg.insert(predictions_copy, write_options={"wait_for_job": True})
            print("Successfully saved predictions to feature store")
            return True
            
        except Exception as e:
            print(f"Save attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return False
            
            import time
            time.sleep(3)
    
    return False


def run_improved_inference_pipeline():
    """Main function to run the improved AQI inference pipeline."""
    print("=" * 80)
    print("IMPROVED LIGHTGBM AQI INFERENCE PIPELINE")
    print("Using Real Forecast Data for Accurate Predictions")
    print("=" * 80)
    
    # Step 1: Connect to Hopsworks
    print("\n[1/8] Connecting to Hopsworks...")
    try:
        project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        print("Successfully connected to Hopsworks")
    except Exception as e:
        print(f"Failed to connect to Hopsworks: {e}")
        return False
    
    # Step 2: Load model
    print("\n[2/8] Loading model from registry...")
    try:
        model, model_features, model_info = load_latest_model_from_registry(mr)
        print(f"Model loaded: {len(model_features)} features")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False
    
    # Step 3: Fetch historical data for lag features
    print("\n[3/8] Fetching historical pollutant data...")
    try:
        df_historical = fetch_historical_pollutants(MAX_LAG_H)
        print("Historical data fetched successfully")
    except Exception as e:
        print(f"Failed to fetch historical data: {e}")
        return False
    
    # Step 4: Fetch real forecast data
    print("\n[4/8] Fetching real pollutant forecasts...")
    try:
        df_forecast = fetch_pollutant_forecasts(HORIZON_H)
        print("Forecast data fetched successfully")
    except Exception as e:
        print(f"Failed to fetch forecast data: {e}")
        return False
    
    # Step 5: Prepare features
    print("\n[5/8] Preparing features with real data...")
    try:
        prediction_data, valid_timestamps = prepare_forecast_features(
            df_historical, df_forecast, model_features
        )
        print("Features prepared successfully")
    except Exception as e:
        print(f"Failed to prepare features: {e}")
        return False
    
    # Step 6: Make predictions
    print("\n[6/8] Making AQI predictions...")
    try:
        predictions_df = make_aqi_predictions(model, prediction_data, valid_timestamps)
        print("Predictions generated successfully")
        
        # Show sample predictions
        print("\nSample predictions:")
        for _, row in predictions_df.head(10).iterrows():
            print(f"   {row['datetime_str']}: AQI {row['predicted_us_aqi']} (Hour +{row['forecast_hour']})")
            
    except Exception as e:
        print(f"Failed to make predictions: {e}")
        return False
    
    # Step 7: Validate predictions
    print("\n[7/8] Validating predictions against API...")
    try:
        validation_metrics = validate_against_api_forecast(predictions_df)
        if validation_metrics:
            print("Validation completed successfully")
        else:
            print("Validation failed, but continuing...")
    except Exception as e:
        print(f"Validation error: {e}")
        validation_metrics = None
    
    # Step 8: Save results
    print("\n[8/8] Saving results...")
    
    # Always save local backup
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"improved_predictions_{timestamp_str}.csv"
    predictions_df.to_csv(backup_path, index=False)
    print(f"Local backup saved: {backup_path}")
    
    # Try to save to feature store
    try:
        success = save_predictions_to_feature_store(predictions_df, fs, model_info)
    except Exception as e:
        print(f"Feature store save failed: {e}")
        success = False
    
    # Final summary
    print("\n" + "=" * 80)
    print("IMPROVED INFERENCE PIPELINE COMPLETED!")
    print("=" * 80)
    print(f"Generated predictions: {len(predictions_df)}")
    print(f"Forecast horizon: {HORIZON_H} hours")
    print(f"AQI range: {predictions_df['predicted_us_aqi'].min()} - {predictions_df['predicted_us_aqi'].max()}")
    print(f"Mean AQI: {predictions_df['predicted_us_aqi'].mean():.1f}")
    print(f"Forecast starts: {valid_timestamps[0].strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Forecast ends: {valid_timestamps[-1].strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Model version: {model_info.get('version', 'unknown')}")
    
    if validation_metrics:
        print(f"Validation MAE: {validation_metrics['mae']:.2f}")
        print(f"Validation Correlation: {validation_metrics['correlation']:.4f}")
    
    print(f"Local backup: {backup_path}")
    print(f"Feature store save: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    """Run the improved inference pipeline."""
    try:
        success = run_improved_inference_pipeline()
        if success:
            print("\n🎉 Pipeline completed successfully with real forecast data!")
        else:
            print("\n⚠️ Pipeline completed with errors!")
            exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        print("\n🏁 Pipeline execution finished")