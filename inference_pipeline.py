# ===================================================================
# 2. OPTIMIZED INFERENCE PIPELINE (inference_pipeline.py)
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
            'HORIZON_H': 74,                     # <-- per your request
            'MAX_LAG_H': 120,
            'MODEL_NAME': "lgb_aqi_forecaster",
            'features': ["pm_10", "pm_25", "carbon_monoxidegm", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
        }

    def create_lag_features(self, df: pd.DataFrame, feat_cols, lags=None):
        if lags is None:
            lags = [1,2,3,6,12,24,48,72,96,120]
        out = df.copy()
        for f in feat_cols:
            for lag in lags:
                out[f"{f}_lag_{lag}"] = out[f].shift(lag)
            out[f"{f}_roll_mean_24"] = out[f].rolling(24, min_periods=24).mean()
            out[f"{f}_roll_std_24"]  = out[f].rolling(24, min_periods=24).std()
            out[f"{f}_roll_mean_72"] = out[f].rolling(72, min_periods=72).mean()
            out[f"{f}_roll_std_72"]  = out[f].rolling(72, min_periods=72).std()
        return out

    def utc_to_tz(self, ts_series: pd.Series, tz: str) -> pd.Series:
        s = pd.to_datetime(ts_series, utc=True)
        # Ensure tz-aware and convert
        if not getattr(s.dt, "tz", None):
            s = s.dt.tz_localize("UTC")
        return s.dt.tz_convert(tz)

    def load_model(self, project):
        """Load latest model from registry (will use v4 if that's the latest)."""
        mr = project.get_model_registry()
        model_meta = mr.get_model(self.config['MODEL_NAME'], version=None)
        model_dir = model_meta.download()
        model = joblib.load(os.path.join(model_dir, "lgb_model.pkl"))
        all_features = joblib.load(os.path.join(model_dir, "lgb_features.pkl"))
        return model, all_features, model_meta.version

    def fetch_forecast_data(self):
        """Fetch pollutant forecast data for next HORIZON_H hours."""
        now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_utc = (now_utc + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_utc = (now_utc + timedelta(hours=self.config['HORIZON_H'])).strftime("%Y-%m-%dT%H:%M:%SZ")
        air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": self.config['LATITUDE'],
            "longitude": self.config['LONGITUDE'],
            "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
            "start": start_utc,
            "end": end_utc,
            "timezone": "UTC",
        }
        resp = requests.get(air_url, params=params)
        resp.raise_for_status()
        raw = resp.json()
        return pd.DataFrame({
            "time_utc": pd.to_datetime(raw["hourly"]["time"]),
            "pm_10": raw["hourly"]["pm10"],
            "pm_25": raw["hourly"]["pm2_5"],
            "carbon_monoxidegm": raw["hourly"]["carbon_monoxide"],
            "nitrogen_dioxide": raw["hourly"]["nitrogen_dioxide"],
            "sulphur_dioxide": raw["hourly"]["sulphur_dioxide"],
            "ozone": raw["hourly"]["ozone"],
        })

    def fetch_historical_data(self):
        """
        Fetch historical data with EXTRA context so lags/rolling features are valid
        for the entire future horizon. We over-fetch by +24h cushion.
        """
        now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        need_ctx = self.config['MAX_LAG_H'] + self.config['HORIZON_H'] + 24
        hist_start = (now_utc - timedelta(hours=need_ctx)).strftime("%Y-%m-%dT%H:%M:%SZ")
        hist_end = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": self.config['LATITUDE'],
            "longitude": self.config['LONGITUDE'],
            "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
            "start": hist_start,
            "end": hist_end,
            "timezone": "UTC",
        }
        resp = requests.get(air_url, params=params)
        resp.raise_for_status()
        raw = resp.json()
        return pd.DataFrame({
            "time_utc": pd.to_datetime(raw["hourly"]["time"]),
            "pm_10": raw["hourly"]["pm10"],
            "pm_25": raw["hourly"]["pm2_5"],
            "carbon_monoxidegm": raw["hourly"]["carbon_monoxide"],
            "nitrogen_dioxide": raw["hourly"]["nitrogen_dioxide"],
            "sulphur_dioxide": raw["hourly"]["sulphur_dioxide"],
            "ozone": raw["hourly"]["ozone"],
        })

    def prepare_features(self, df_hist, df_future, all_features):
        """
        Prepare features for prediction.
        If history is too short, we will still build what we can; later we’ll
        align df_future to the number of rows that actually have complete features.
        """
        base_cols = self.config['features']
        hist = df_hist[base_cols].copy().reset_index(drop=True)
        fut  = df_future[base_cols].copy().reset_index(drop=True)

        combined = pd.concat([hist, fut], ignore_index=True)
        combined = self.create_lag_features(combined, base_cols)
        combined = combined.dropna().reset_index(drop=True)

        # For a full horizon we expect len(df_future) rows at the tail.
        # If history was insufficient, this may be smaller.
        expected = len(df_future)
        future_block = combined.tail(min(expected, len(combined))).copy()

        # Ensure all features required by the model exist
        for c in all_features:
            if c not in future_block.columns:
                future_block[c] = 0.0

        # Return strictly in the model feature order
        future_block = future_block[all_features]
        return future_block

    def run_inference(self):
        """Main inference pipeline"""
        try:
            logger.info("Starting inference pipeline...")

            # Connect to Hopsworks
            project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
            fs = project.get_feature_store()

            # Load model
            model, all_features, model_version = self.load_model(project)

            # Fetch data
            df_future = self.fetch_forecast_data()
            df_hist = self.fetch_historical_data()

            # Prepare features
            X_future = self.prepare_features(df_hist, df_future, all_features)

            # Make predictions
            predictions = model.predict(X_future)
            n_pred = len(predictions)

            if n_pred < len(df_future):
                logger.warning(
                    f"Only {n_pred} / {len(df_future)} future rows have complete lag/rolling features "
                    f"(likely due to limited historical data from API). "
                    f"Slicing future timestamps to match."
                )

            # Align df_future to number of predictions to avoid length mismatch
            df_future_aligned = df_future.tail(n_pred).copy().reset_index(drop=True)

            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                "datetime": self.utc_to_tz(df_future_aligned["time_utc"], self.config['TZ']),
                "datetime_utc": df_future_aligned["time_utc"],
                "predicted_us_aqi": predictions,
            })

            # Add metadata
            forecast_df["prediction_date"] = datetime.now(timezone.utc)
            forecast_df["model_version"] = model_version

            # === Save to Feature Store ===
            predictions_fg = fs.get_or_create_feature_group(
                name="aqi_predictions",
                version=1,
                description="Hourly AQI forecasts with model version and prediction timestamp",
                primary_key=["datetime_utc", "prediction_date"],
                event_time="datetime_utc",
            )
            predictions_fg.insert(forecast_df)

            # Save also as CSV for backup
            forecast_path = f"aqi_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            forecast_df.to_csv(forecast_path, index=False)

            result = {
                "status": "success",
                "model_version": model_version,
                "forecast_file": forecast_path,
                "predictions_count": int(n_pred),
                "avg_aqi": float(np.mean(predictions)) if n_pred > 0 else None,
            }

            logger.info(f"Inference completed: {n_pred} predictions stored in feature store")
            return result

        except Exception as e:
            logger.error(f"Inference pipeline failed: {e}")
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    pipeline = AQIInferencePipeline()
    result = pipeline.run_inference()
    print(result)
