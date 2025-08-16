# ===================================================================
# 1. OPTIMIZED TRAINING PIPELINE (training_pipeline.py)
# ===================================================================

import os
import sys
import logging
import traceback
from datetime import datetime, timezone
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import hopsworks
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hsml.model import ModelSchema, Schema

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
API_KEY = os.environ["HOPSWORKS_API_KEY"]

class AQITrainingPipeline:
    def __init__(self):
        self.config = {
            'FEATURE_GROUP_NAME': "aqi_weather_features",
            'FEATURE_GROUP_VER': 2,
            'MODEL_NAME': "lgb_aqi_forecaster",
            'ARTIFACT_DIR': "lgb_aqi_artifacts",
            'MAX_LAG_H': 120,
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

    def load_existing_model(self, project):
        """Load existing model from registry or return None for first training"""
        try:
            mr = project.get_model_registry()
            model_meta = mr.get_model(self.config['MODEL_NAME'], version=None)
            model_dir = model_meta.download()
            
            existing_model = joblib.load(os.path.join(model_dir, "lgb_model.pkl"))
            all_features = joblib.load(os.path.join(model_dir, "lgb_features.pkl"))
            last_trained_time = joblib.load(os.path.join(model_dir, "last_trained_timestamp.pkl"))
            
            return existing_model, all_features, last_trained_time, model_meta.version
        except Exception as e:
            logger.info(f"No existing model found, starting fresh training: {e}")
            return None, None, None, None

    def prepare_training_data(self, df, last_trained_time=None):
        """Prepare training data (incremental if last_trained_time provided)"""
        df["time_utc"] = self.ensure_utc(df["time"])
        
        if last_trained_time:
            # Incremental training - get new data + context for lag features
            context_start = last_trained_time - pd.Timedelta(hours=self.config['MAX_LAG_H'])
            context_mask = df["time_utc"] >= context_start
            df_context = df[context_mask].copy()
            
            # Create features for context + new data
            work = df_context.set_index("time_utc")[self.config['features'] + ["us_aqi"]].copy()
            work = self.create_lag_features(work, self.config['features'])
            work.dropna(inplace=True)
            
            # Extract only new training examples
            new_training_mask = work.index > last_trained_time
            work_new = work[new_training_mask].copy()
            
            if len(work_new) == 0:
                return None, None, None, True  # No new data
                
            return work_new, work, None, False
        else:
            # Fresh training - use all data
            work = df.set_index("time_utc")[self.config['features'] + ["us_aqi"]].copy()
            work = self.create_lag_features(work, self.config['features'])
            work.dropna(inplace=True)
            
            # Train/test split for fresh training
            split_idx = int(len(work) * 0.8)
            train_data = work.iloc[:split_idx]
            test_data = work.iloc[split_idx:]
            
            return train_data, work, test_data, False

    def train_model(self, train_data, existing_model=None, is_incremental=False):
        """Train LightGBM model"""
        all_features = [c for c in train_data.columns if c != "us_aqi"]
        X_train = train_data[all_features]
        y_train = train_data["us_aqi"]
        
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
        
        if is_incremental and existing_model:
            # Incremental training
            additional_rounds = min(200, max(50, len(X_train) // 10))
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=additional_rounds,
                init_model=existing_model,
                callbacks=[lgb.log_evaluation(0)]
            )
            logger.info(f"Incremental training with {additional_rounds} additional rounds")
        else:
            # Fresh training
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=800,
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
            )
            logger.info("Fresh training completed")
            
        return model, all_features

    def evaluate_model(self, model, test_data, all_features):
        """Evaluate model performance"""
        if test_data is None or len(test_data) == 0:
            return None, None, None
            
        X_test = test_data[all_features]
        y_test = test_data["us_aqi"]
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return mae, rmse, r2

    def save_model_to_registry(self, project, model, all_features, last_trained_time, metrics):
        """Save model to Hopsworks registry"""
        try:
            # Create dummy data for schema
            dummy_X = pd.DataFrame(np.zeros((1, len(all_features))), columns=all_features)
            dummy_y = pd.Series([0.0])
            
            input_schema = Schema(dummy_X)
            output_schema = Schema(dummy_y)
            model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
            
            mr = project.get_model_registry()
            model_meta = mr.python.create_model(
                name=self.config['MODEL_NAME'],
                metrics=metrics or {},
                model_schema=model_schema,
                description=f"LightGBM AQI forecaster - Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # Save artifacts
            os.makedirs(self.config['ARTIFACT_DIR'], exist_ok=True)
            joblib.dump(model, os.path.join(self.config['ARTIFACT_DIR'], "lgb_model.pkl"))
            joblib.dump(all_features, os.path.join(self.config['ARTIFACT_DIR'], "lgb_features.pkl"))
            joblib.dump(last_trained_time, os.path.join(self.config['ARTIFACT_DIR'], "last_trained_timestamp.pkl"))
            
            model_meta.save(self.config['ARTIFACT_DIR'])
            logger.info("Model saved to registry successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model to registry: {e}")
            return False

    def run_pipeline(self):
        """Main training pipeline"""
        try:
            logger.info("Starting training pipeline...")
            
            # Connect to Hopsworks
            # project = hopsworks.login()
            project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")

            fs = project.get_feature_store()
            fg = fs.get_feature_group(name=self.config['FEATURE_GROUP_NAME'], version=self.config['FEATURE_GROUP_VER'])
            
            # Load data
            df_raw = fg.read()
            df_raw = df_raw.sort_values("time", ascending=True).reset_index(drop=True)
            
            cols_needed = ["time", "pm_10", "pm_25", "carbon_monoxidegm", "nitrogen_dioxide",
                          "sulphur_dioxide", "ozone", "us_aqi"]
            df = df_raw[cols_needed].copy()
            
            # Load existing model if available
            existing_model, existing_features, last_trained_time, model_version = self.load_existing_model(project)
            
            # Prepare training data
            train_data, full_work_data, test_data, no_new_data = self.prepare_training_data(df, last_trained_time)
            
            if no_new_data:
                logger.info("No new data available for training. Exiting.")
                return {"status": "no_new_data"}
            
            # Train model
            is_incremental = existing_model is not None
            model, all_features = self.train_model(train_data, existing_model, is_incremental)
            
            # Evaluate model
            mae, rmse, r2 = self.evaluate_model(model, test_data or train_data, all_features)
            metrics = {"mae": mae, "rmse": rmse, "r2": r2} if mae is not None else {}
            
            # Update last trained timestamp
            new_last_trained_time = train_data.index.max() if train_data is not None else full_work_data.index.max()
            
            # Save to registry
            success = self.save_model_to_registry(project, model, all_features, new_last_trained_time, metrics)
            
            result = {
                "status": "success",
                "training_type": "incremental" if is_incremental else "fresh",
                "metrics": metrics,
                "last_trained_time": str(new_last_trained_time),
                "registry_saved": success
            }
            
            logger.info(f"Training completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}
        
if __name__ == "__main__":
    pipeline = AQITrainingPipeline()
    result = pipeline.run_pipeline()
    print(result)
