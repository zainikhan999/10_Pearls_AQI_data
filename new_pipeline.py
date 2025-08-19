# ===================================================================
# FIXED AQI PIPELINE - 72 HOURS PREDICTION WITH AUTO-RETRAINING
# ===================================================================

import requests
from datetime import datetime, timedelta, timezone
import os
import joblib
import numpy as np
import pandas as pd
import hopsworks
import traceback
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
API_KEY = os.environ["HOPSWORKS_API_KEY"]

class FixedAQIPipeline:
    def __init__(self):
        self.config = {
            'LATITUDE': 33.5973,
            'LONGITUDE': 73.0479,
            'TZ': "Asia/Karachi",
            'HORIZON_H': 72,  # 3 days prediction
            'MAX_LAG_H': 120,
            'FEATURE_GROUP_NAME': "aqi_weather_features",
            'FEATURE_GROUP_VER': 2,
            'MODEL_NAME': "lgb_aqi_forecaster",
            'ARTIFACT_DIR': "lgb_aqi_artifacts",
            'features': ["pm_10", "pm_25", "carbon_monoxidegm", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
        }

    def create_lag_features(self, df: pd.DataFrame, feat_cols, lags=None):
        if lags is None:
            lags = [1,2,3,6,12,24,48,72,96,120]
        out = df.copy()
        for f in feat_cols:
            for lag in lags:
                out[f"{f}_lag_{lag}"] = out[f].shift(lag)
            out[f"{f}_roll_mean_24"] = out[f].rolling(24, min_periods=12).mean()
            out[f"{f}_roll_std_24"]  = out[f].rolling(24, min_periods=12).std()
            out[f"{f}_roll_mean_72"] = out[f].rolling(72, min_periods=36).mean()
            out[f"{f}_roll_std_72"]  = out[f].rolling(72, min_periods=36).std()
        return out

    def utc_to_tz(self, ts_series: pd.Series, tz: str) -> pd.Series:
        s = pd.to_datetime(ts_series, utc=True)
        return s.dt.tz_convert(tz)

    def ensure_utc(self, ts_series: pd.Series) -> pd.Series:
        s = pd.to_datetime(ts_series)
        try:
            if s.dt.tz is None:
                return s.dt.tz_localize("UTC")
            else:
                return s.dt.tz_convert("UTC")
        except AttributeError:
            s = pd.to_datetime(s, errors="coerce")
            s = s.dt.tz_localize("UTC")
            return s

    def get_last_feature_store_timestamp(self, fs):
        """Get the last timestamp from feature store"""
        try:
            fg = fs.get_feature_group(name=self.config['FEATURE_GROUP_NAME'], version=self.config['FEATURE_GROUP_VER'])
            # Get latest records to find last timestamp
            latest_data = fg.select(["time"]).read()
            if len(latest_data) > 0:
                latest_data["time_utc"] = self.ensure_utc(latest_data["time"])
                last_timestamp = latest_data["time_utc"].max()
                logger.info(f"Last timestamp in feature store: {last_timestamp}")
                return last_timestamp
            else:
                logger.warning("No data in feature store")
                return None
        except Exception as e:
            logger.error(f"Error getting last timestamp: {e}")
            return None

    def fetch_historical_data_from_api(self, start_time, end_time):
        """Fetch historical data from API to fill gaps"""
        air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": self.config['LATITUDE'],
            "longitude": self.config['LONGITUDE'],
            "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
            "start": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timezone": "UTC",
        }
        
        resp = requests.get(air_url, params=params)
        resp.raise_for_status()
        raw = resp.json()
        
        return pd.DataFrame({
            "time": pd.to_datetime(raw["hourly"]["time"]),
            "pm_10": raw["hourly"]["pm10"],
            "pm_25": raw["hourly"]["pm2_5"],
            "carbon_monoxidegm": raw["hourly"]["carbon_monoxide"],
            "nitrogen_dioxide": raw["hourly"]["nitrogen_dioxide"],
            "sulphur_dioxide": raw["hourly"]["sulphur_dioxide"],
            "ozone": raw["hourly"]["ozone"],
        })

    def get_complete_historical_data(self, project, min_hours_needed=200):
        """Get complete historical data from feature store + API if needed"""
        fs = project.get_feature_store()
        
        # Get data from feature store
        try:
            fg = fs.get_feature_group(name=self.config['FEATURE_GROUP_NAME'], version=self.config['FEATURE_GROUP_VER'])
            df_fs = fg.read()
            df_fs = df_fs.sort_values("time", ascending=True).reset_index(drop=True)
            logger.info(f"Loaded {len(df_fs)} records from feature store")
        except Exception as e:
            logger.error(f"Error loading from feature store: {e}")
            df_fs = pd.DataFrame()

        # Get last timestamp from feature store
        now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        
        if len(df_fs) > 0:
            df_fs["time_utc"] = self.ensure_utc(df_fs["time"])
            last_fs_time = df_fs["time_utc"].max()
            
            # Check if we need more recent data
            hours_since_last = (now_utc - last_fs_time).total_seconds() / 3600
            
            if hours_since_last > 1:  # More than 1 hour gap
                logger.info(f"Gap of {hours_since_last:.1f} hours detected. Fetching recent data from API...")
                
                # Fetch recent data from API
                api_start = last_fs_time + timedelta(hours=1)
                recent_data = self.fetch_historical_data_from_api(api_start, now_utc)
                
                if len(recent_data) > 0:
                    # Add missing US AQI column (we'll estimate it)
                    recent_data["us_aqi"] = np.maximum(
                        recent_data["pm_25"] * 2.5,  # Rough PM2.5 to AQI conversion
                        recent_data["pm_10"] * 1.5   # Rough PM10 to AQI conversion
                    )
                    
                    # Combine feature store data with recent API data
                    cols_needed = ["time", "pm_10", "pm_25", "carbon_monoxidegm", 
                                 "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]
                    
                    df_fs_subset = df_fs[cols_needed].copy() if len(df_fs) > 0 else pd.DataFrame(columns=cols_needed)
                    recent_data_subset = recent_data[cols_needed[:-1]].copy()  # No us_aqi in API data initially
                    recent_data_subset["us_aqi"] = recent_data["us_aqi"]
                    
                    combined_data = pd.concat([df_fs_subset, recent_data_subset], ignore_index=True)
                    combined_data = combined_data.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
                    
                    logger.info(f"Combined data: {len(combined_data)} records total")
                    return combined_data
        
        # If we have enough data from feature store, use it
        if len(df_fs) >= min_hours_needed:
            cols_needed = ["time", "pm_10", "pm_25", "carbon_monoxidegm", 
                          "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]
            return df_fs[cols_needed].tail(min_hours_needed * 2)  # Take recent data
        
        # If not enough data, fetch from API
        logger.info("Insufficient data in feature store. Fetching from API...")
        api_start = now_utc - timedelta(hours=min_hours_needed)
        api_data = self.fetch_historical_data_from_api(api_start, now_utc)
        
        # Estimate US AQI for API data
        api_data["us_aqi"] = np.maximum(
            api_data["pm_25"] * 2.5,
            api_data["pm_10"] * 1.5
        )
        
        return api_data

    def retrain_model(self, df_historical):
        """Retrain model on historical data"""
        logger.info("Starting model retraining...")
        
        # Prepare training data
        df_historical["time_utc"] = self.ensure_utc(df_historical["time"])
        work = df_historical.set_index("time_utc")[self.config['features'] + ["us_aqi"]].copy()
        work = self.create_lag_features(work, self.config['features'])
        work.dropna(inplace=True)
        
        if len(work) < 100:
            raise ValueError(f"Not enough training data after feature engineering: {len(work)} samples")
        
        # Train/test split
        split_idx = int(len(work) * 0.85)
        train_data = work.iloc[:split_idx]
        test_data = work.iloc[split_idx:]
        
        # Prepare features
        all_features = [c for c in train_data.columns if c != "us_aqi"]
        X_train = train_data[all_features]
        y_train = train_data["us_aqi"]
        X_test = test_data[all_features]
        y_test = test_data["us_aqi"]
        
        # Train model
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        lgb_train = lgb.Dataset(X_train, y_train)
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model retrained - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")
        
        return model, all_features, {"mae": mae, "rmse": rmse, "r2": r2}

    def fetch_forecast_data(self):
        """Fetch forecast data for next 72 hours"""
        now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_utc = (now_utc + timedelta(hours=1))
        end_utc = (now_utc + timedelta(hours=self.config['HORIZON_H']))
        
        air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": self.config['LATITUDE'],
            "longitude": self.config['LONGITUDE'],
            "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
            "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
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

    def prepare_forecast_features(self, df_hist, df_future, all_features):
        """Prepare features for 72-hour forecast"""
        logger.info(f"Preparing features: {len(df_hist)} historical + {len(df_future)} future points")
        
        # Ensure we have enough historical context
        df_hist = df_hist.tail(self.config['MAX_LAG_H'] * 2)  # Keep more history for robust lag features
        
        # Combine historical and future data
        hist_data = df_hist[self.config['features']].copy()
        future_data = df_future[self.config['features']].copy()
        
        # Create time index for proper alignment
        hist_times = pd.date_range(
            end=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0),
            periods=len(hist_data),
            freq='H'
        )
        future_times = pd.date_range(
            start=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1),
            periods=len(future_data),
            freq='H'
        )
        
        # Combine with proper time index
        combined_data = pd.concat([
            pd.DataFrame(hist_data.values, index=hist_times, columns=self.config['features']),
            pd.DataFrame(future_data.values, index=future_times, columns=self.config['features'])
        ])
        
        # Create lag features
        combined_with_lags = self.create_lag_features(combined_data, self.config['features'])
        
        # Extract future portion with valid lag features
        future_start_idx = len(hist_data)
        future_with_lags = combined_with_lags.iloc[future_start_idx:].copy()
        
        # Drop rows with too many NaNs (but be more lenient)
        future_with_lags = future_with_lags.dropna(thresh=len(future_with_lags.columns) * 0.7)  # Allow 30% NaNs
        
        # Fill remaining NaNs with forward fill and then backward fill
        future_with_lags = future_with_lags.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure all required features are present
        for feature in all_features:
            if feature not in future_with_lags.columns:
                future_with_lags[feature] = 0.0
        
        logger.info(f"Final forecast features prepared: {len(future_with_lags)} time points")
        
        return future_with_lags[all_features], future_with_lags.index

    def run_complete_pipeline(self):
        """Complete pipeline: retrain model + make 72-hour predictions"""
        try:
            logger.info("Starting complete AQI pipeline...")
            
            # Connect to Hopsworks
            project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
            fs = project.get_feature_store()
            
            # Get complete historical data (feature store + recent API data)
            df_historical = self.get_complete_historical_data(project, min_hours_needed=300)
            logger.info(f"Historical data loaded: {len(df_historical)} records")
            
            # Retrain model on complete historical data
            model, all_features, metrics = self.retrain_model(df_historical)
            
            # Fetch forecast data for next 72 hours
            df_future = self.fetch_forecast_data()
            logger.info(f"Forecast data fetched: {len(df_future)} hours")
            
            # Prepare features for prediction
            X_future, future_times = self.prepare_forecast_features(df_historical, df_future, all_features)
            
            # Make predictions
            predictions = model.predict(X_future)
            logger.info(f"Predictions generated: {len(predictions)} time points")
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                "datetime": self.utc_to_tz(pd.Series(future_times), self.config['TZ']),
                "datetime_utc": future_times,
                "predicted_us_aqi": predictions,
                "prediction_date": datetime.now(timezone.utc),
                "model_version": "retrained_" + datetime.now().strftime("%Y%m%d_%H%M")
            })
            
            # Save predictions to feature store
            predictions_fg = fs.get_or_create_feature_group(
                name="aqi_predictions",
                version=1,
                description="72-hour AQI forecasts with retrained model",
                primary_key=["datetime_utc", "prediction_date"],
                event_time="datetime_utc",
            )
            predictions_fg.insert(forecast_df)
            
            # Save as CSV backup
            forecast_path = f"aqi_forecast_72h_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            forecast_df.to_csv(forecast_path, index=False)
            
            result = {
                "status": "success",
                "model_metrics": metrics,
                "predictions_count": len(predictions),
                "forecast_hours": len(predictions),
                "avg_aqi": np.mean(predictions),
                "max_aqi": np.max(predictions),
                "min_aqi": np.min(predictions),
                "forecast_file": forecast_path,
                "forecast_data": forecast_df.to_dict('records')[:10]  # First 10 predictions
            }
            
            logger.info(f"✅ PIPELINE SUCCESS: {len(predictions)} hours of AQI predictions generated")
            logger.info(f"📊 Average AQI: {np.mean(predictions):.1f}")
            logger.info(f"📈 AQI Range: {np.min(predictions):.1f} - {np.max(predictions):.1f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ PIPELINE FAILED: {e}")
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    pipeline = FixedAQIPipeline()
    result = pipeline.run_complete_pipeline()
    print("\n" + "="*50)
    print("FINAL RESULT:")
    print("="*50)
    for key, value in result.items():
        if key != "forecast_data":
            print(f"{key}: {value}")
    print("="*50)