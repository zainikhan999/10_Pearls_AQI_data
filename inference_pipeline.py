# ===================================================================
# FIXED INFERENCE PIPELINE (inference_pipeline.py)
# ===================================================================

import requests
from datetime import datetime, timedelta, timezone
import os
import joblib
import numpy as np
import pandas as pd
import hopsworks
import traceback
import logging
from pytz import timezone as pytz_timezone

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
API_KEY = os.environ["HOPSWORKS_API_KEY"]

class AQIInferencePipeline:
    def __init__(self):
        self.config = {
            'LATITUDE': 33.5973,
            'LONGITUDE': 73.0479,
            'TZ': "Asia/Karachi",
            'HORIZON_H': 74,
            'MAX_LAG_H': 120,
            'MODEL_NAME': "lgb_aqi_forecaster",
            'FEATURE_GROUP_NAME': "aqi_weather_features",
            'FEATURE_GROUP_VER': 2,
            'features': ["pm_10", "pm_25", "carbon_monoxidegm", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
        }

    def ensure_utc(self, time_series):
        """Ensure datetime series is in UTC"""
        if time_series.dt.tz is None:
            return pd.to_datetime(time_series, utc=True)
        else:
            return time_series.dt.tz_convert('UTC')

    def utc_to_tz(self, utc_series, target_tz):
        """Convert UTC datetime series to target timezone"""
        return utc_series.dt.tz_convert(target_tz)

    def create_lag_features(self, df, features, max_lag=120):
        """Create lag features for time series prediction"""
        df_lagged = df.copy()
        
        # Create lag features
        for feature in features:
            if feature in df.columns:
                for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
                    if lag <= max_lag:
                        df_lagged[f"{feature}_lag{lag}"] = df[feature].shift(lag)
        
        # Add moving averages
        for feature in features:
            if feature in df.columns:
                df_lagged[f"{feature}_ma24"] = df[feature].rolling(window=24, min_periods=12).mean()
                df_lagged[f"{feature}_ma72"] = df[feature].rolling(window=72, min_periods=36).mean()
        
        # Add time features
        df_lagged['hour'] = df_lagged.index.hour
        df_lagged['day_of_week'] = df_lagged.index.dayofweek
        df_lagged['month'] = df_lagged.index.month
        
        return df_lagged

    def load_model(self, project):
        """Load model from Hopsworks registry"""
        try:
            mr = project.get_model_registry()
            model_meta = mr.get_model(self.config['MODEL_NAME'], version=None)
            model_dir = model_meta.download()
            
            model = joblib.load(os.path.join(model_dir, "lgb_model.pkl"))
            all_features = joblib.load(os.path.join(model_dir, "lgb_features.pkl"))
            
            return model, all_features, model_meta.version
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def fetch_historical_data(self, fs):
        """Fetch sufficient historical data for lag features"""
        try:
            fg = fs.get_feature_group(name=self.config['FEATURE_GROUP_NAME'], version=self.config['FEATURE_GROUP_VER'])
            df_hist = fg.read()
            
            # Ensure UTC timezone
            df_hist["time_utc"] = self.ensure_utc(df_hist["time"])
            df_hist = df_hist.sort_values("time_utc").reset_index(drop=True)
            
            # Keep only last MAX_LAG_H + buffer hours for efficiency
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.config['MAX_LAG_H'] + 24)
            df_hist = df_hist[df_hist['time_utc'] >= cutoff_time]
            
            logger.info(f"Loaded {len(df_hist)} historical rows from {df_hist['time_utc'].min()} to {df_hist['time_utc'].max()}")
            return df_hist
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            raise

    def determine_prediction_start_time(self, fs):
        """
        FIXED: Determine where to start predictions from based on:
        1. Latest available feature data
        2. Existing predictions
        """
        
        # Get latest feature data timestamp
        try:
            fg = fs.get_feature_group(name=self.config['FEATURE_GROUP_NAME'], version=self.config['FEATURE_GROUP_VER'])
            feature_data = fg.read()
            
            if feature_data.empty:
                raise ValueError("No feature data available")
            
            feature_data["time_utc"] = self.ensure_utc(feature_data["time"])
            latest_feature_time = feature_data["time_utc"].max()
            logger.info(f"📊 Latest feature data: {latest_feature_time}")
            
        except Exception as e:
            logger.error(f"Failed to get latest feature time: {e}")
            raise
        
        # Check existing predictions
        try:
            predictions_fg = fs.get_feature_group("aqi_predictions", version=1)
            existing_predictions = predictions_fg.read()
            
            if not existing_predictions.empty:
                existing_predictions["datetime_utc"] = pd.to_datetime(existing_predictions["datetime_utc"], utc=True)
                latest_prediction_time = existing_predictions["datetime_utc"].max()
                logger.info(f"🔮 Latest prediction: {latest_prediction_time}")
                
                # Start from the later of: next hour after latest prediction OR next hour after latest feature data
                start_from_predictions = latest_prediction_time + timedelta(hours=1)
                start_from_features = latest_feature_time + timedelta(hours=1)
                
                start_time = max(start_from_predictions, start_from_features)
                logger.info(f"🎯 Starting predictions from: {start_time}")
                
            else:
                # No predictions exist - start from next hour after latest feature data
                start_time = latest_feature_time + timedelta(hours=1)
                logger.info(f"🆕 No existing predictions, starting from: {start_time}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not check existing predictions: {e}")
            # Fallback: start from next hour after latest feature data
            start_time = latest_feature_time + timedelta(hours=1)
        
        return start_time, latest_feature_time

    def fetch_forecast_data(self, fs):
        """Fetch forecast data for predictions"""
        
        start_time, latest_feature_time = self.determine_prediction_start_time(fs)
        
        # Generate predictions up to HORIZON_H hours in the future from NOW
        now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        end_time = now_utc + timedelta(hours=self.config['HORIZON_H'])
        
        # If start_time is in the past, we can backfill some predictions
        # But don't go beyond latest_feature_time + HORIZON_H
        max_end_time = latest_feature_time + timedelta(hours=self.config['HORIZON_H'])
        end_time = min(end_time, max_end_time)
        
        if start_time >= end_time:
            logger.info("✅ All predictions up to horizon are already available")
            return pd.DataFrame()
        
        logger.info(f"🎯 Fetching forecast data from {start_time} to {end_time} ({(end_time-start_time).total_seconds()/3600:.0f} hours)")
        
        air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": self.config['LATITUDE'],
            "longitude": self.config['LONGITUDE'],
            "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
            "start": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timezone": "UTC",
        }
        
        try:
            resp = requests.get(air_url, params=params, timeout=30)
            resp.raise_for_status()
            raw = resp.json()
            
            df = pd.DataFrame({
                "time_utc": pd.to_datetime(raw["hourly"]["time"]),
                "pm_10": raw["hourly"]["pm10"],
                "pm_25": raw["hourly"]["pm2_5"],
                "carbon_monoxidegm": raw["hourly"]["carbon_monoxide"],
                "nitrogen_dioxide": raw["hourly"]["nitrogen_dioxide"],
                "sulphur_dioxide": raw["hourly"]["sulphur_dioxide"],
                "ozone": raw["hourly"]["ozone"],
            })
            
            logger.info(f"📈 Retrieved {len(df)} hours of forecast data")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch forecast data: {e}")
            raise

    def prepare_features(self, df_hist, df_future, all_features):
        """Prepare features for prediction"""
        
        # Combine historical and future data
        cols_needed = ["time_utc"] + self.config['features']
        
        # Historical data (has target values)
        hist_work = df_hist[cols_needed + ["us_aqi"]].copy()
        hist_work = hist_work.set_index("time_utc")
        
        # Future data (no target values - we're predicting these)
        future_work = df_future[cols_needed].copy()
        future_work = future_work.set_index("time_utc")
        future_work["us_aqi"] = np.nan  # We don't know these yet
        
        # Combine for feature creation
        combined_work = pd.concat([hist_work, future_work]).sort_index()
        
        # Create lag features
        combined_with_lags = self.create_lag_features(combined_work, self.config['features'])
        
        # Extract only future data with features
        future_with_features = combined_with_lags.loc[future_work.index]
        
        # Keep only model features and drop rows with missing lag features
        X_future = future_with_features[all_features].dropna()
        
        logger.info(f"🔧 Prepared features for {len(X_future)} prediction timestamps")
        
        return X_future

    def run_inference(self):
        """Main inference pipeline"""
        try:
            logger.info("🚀 Starting AQI inference pipeline...")

            # Connect to Hopsworks
            project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
            fs = project.get_feature_store()

            # Load model
            model, all_features, model_version = self.load_model(project)
            logger.info(f"📦 Loaded model version {model_version}")

            # Fetch data for predictions
            df_future = self.fetch_forecast_data(fs)
            
            if df_future.empty:
                logger.info("✅ No new predictions needed - all up to date!")
                return {"status": "no_new_predictions_needed"}

            # Fetch historical data for lag features  
            df_hist = self.fetch_historical_data(fs)

            # Prepare features
            X_future = self.prepare_features(df_hist, df_future, all_features)
            
            if len(X_future) == 0:
                logger.warning("⚠️ No valid features for prediction (possibly insufficient historical data)")
                return {"status": "no_valid_features"}

            # Make predictions
            predictions = model.predict(X_future)
            n_pred = len(predictions)

            logger.info(f"🎯 Generated {n_pred} predictions")

            # Align df_future to predictions
            prediction_timestamps = X_future.index
            df_future_aligned = df_future[df_future['time_utc'].isin(prediction_timestamps)].copy()

            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                "datetime": self.utc_to_tz(pd.to_datetime(prediction_timestamps), self.config['TZ']),
                "datetime_utc": prediction_timestamps,
                "predicted_us_aqi": predictions,
            })

            # Add metadata
            forecast_df["prediction_date"] = datetime.now(timezone.utc)
            forecast_df["model_version"] = model_version

            # Save to Feature Store
            try:
                predictions_fg = fs.get_or_create_feature_group(
                    name="aqi_predictions",
                    version=1,
                    description="Hourly AQI forecasts with model version and prediction timestamp",
                    primary_key=["datetime_utc", "prediction_date"],
                    event_time="datetime_utc",
                )
                predictions_fg.insert(forecast_df)
                logger.info("💾 Saved predictions to feature store")

            except Exception as e:
                logger.error(f"Failed to save predictions: {e}")
                # Continue anyway - return results

            result = {
                "status": "success",
                "model_version": model_version,
                "predictions_count": int(n_pred),
                "avg_aqi": float(np.mean(predictions)) if n_pred > 0 else None,
                "time_range": f"{prediction_timestamps.min()} to {prediction_timestamps.max()}",
                "prediction_hours": int(n_pred)
            }

            logger.info(f"✅ Inference completed: {n_pred} predictions generated")
            return result

        except Exception as e:
            logger.error(f"❌ Inference pipeline failed: {e}")
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    pipeline = AQIInferencePipeline()
    result = pipeline.run_inference()
    print(result)