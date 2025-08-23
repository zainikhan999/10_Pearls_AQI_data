# ===================================================================
# FIXED TRAINING PIPELINE (training_pipeline.py)
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

    def ensure_utc(self, time_series):
        """Ensure datetime series is in UTC"""
        if time_series.dt.tz is None:
            return pd.to_datetime(time_series, utc=True)
        else:
            return time_series.dt.tz_convert('UTC')

    def create_lag_features(self, df, features, max_lag=120):
        """Create lag features for time series prediction"""
        logger.info(f"Creating lag features for {len(features)} features with max lag {max_lag}")
        
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

    def load_existing_model_and_info(self, project):
        """Load existing model and training metadata"""
        try:
            mr = project.get_model_registry()
            model_meta = mr.get_model(self.config['MODEL_NAME'], version=None)
            model_dir = model_meta.download()
            
            existing_model = joblib.load(os.path.join(model_dir, "lgb_model.pkl"))
            all_features = joblib.load(os.path.join(model_dir, "lgb_features.pkl"))
            
            # Load training metadata
            try:
                training_info = joblib.load(os.path.join(model_dir, "training_info.pkl"))
            except:
                # Legacy model - create default info
                training_info = {
                    "last_training_time": None, 
                    "total_samples": 0,
                    "last_data_timestamp": None
                }
            
            return existing_model, all_features, training_info, model_meta.version
        except Exception as e:
            logger.info(f"No existing model found, starting fresh training: {e}")
            return None, None, {"last_training_time": None, "total_samples": 0, "last_data_timestamp": None}, None

    def prepare_training_data(self, df_raw, training_info, existing_model=None):
        """Prepare training data - only train on new data since last training"""
        
        # Ensure time column is UTC
        df_raw["time_utc"] = self.ensure_utc(df_raw["time"])
        df_raw = df_raw.sort_values("time_utc").reset_index(drop=True)
        
        logger.info(f"Total rows in feature store: {len(df_raw)}")
        logger.info(f"Date range: {df_raw['time_utc'].min()} to {df_raw['time_utc'].max()}")
        
        # Determine what data is new
        if existing_model is not None and training_info.get('last_data_timestamp'):
            last_trained_timestamp = pd.to_datetime(training_info['last_data_timestamp'], utc=True)
            logger.info(f"Last training used data up to: {last_trained_timestamp}")
            
            # Find new data (after last training timestamp)
            new_data_mask = df_raw['time_utc'] > last_trained_timestamp
            new_data_count = new_data_mask.sum()
            
            if new_data_count == 0:
                logger.info("No new data available for training")
                return None, None, None, True
            
            logger.info(f"Found {new_data_count} new rows for training")
            
            # For incremental training, we need historical context for lag features
            # Use last MAX_LAG_H hours before new data for context
            context_start_time = df_raw[new_data_mask]['time_utc'].min() - pd.Timedelta(hours=self.config['MAX_LAG_H'])
            context_mask = df_raw['time_utc'] >= context_start_time
            
            # Prepare data with context
            context_df = df_raw[context_mask].copy()
            cols_needed = ["time_utc"] + self.config['features'] + ["us_aqi"]
            work_df = context_df[cols_needed].set_index("time_utc")
            
            # Create lag features
            work_df = self.create_lag_features(work_df, self.config['features'])
            work_df.dropna(inplace=True)
            
            # Extract only NEW data for training
            new_training_data = work_df[work_df.index > last_trained_timestamp]
            
            if len(new_training_data) == 0:
                logger.warning("No new training data after creating lag features")
                return None, None, None, True
            
            logger.info(f"Incremental training: {len(new_training_data)} new samples")
            latest_data_timestamp = df_raw['time_utc'].max()
            
            return new_training_data, work_df, None, False, latest_data_timestamp
        
        else:
            # Fresh training - use all data
            cols_needed = ["time_utc"] + self.config['features'] + ["us_aqi"]
            work_df = df_raw[cols_needed].set_index("time_utc")
            work_df = self.create_lag_features(work_df, self.config['features'])
            work_df.dropna(inplace=True)
            
            # Train/test split for fresh training
            split_idx = int(len(work_df) * 0.8)
            train_data = work_df.iloc[:split_idx]
            test_data = work_df.iloc[split_idx:]
            
            logger.info(f"Fresh training: {len(train_data)} training samples, {len(test_data)} test samples")
            latest_data_timestamp = df_raw['time_utc'].max()
            
            return train_data, work_df, test_data, False, latest_data_timestamp

    def train_model(self, train_data, existing_model=None, is_incremental=False):
        """Train or retrain the LightGBM model"""
        
        # Prepare features and target
        feature_cols = [col for col in train_data.columns if col != 'us_aqi']
        X_train = train_data[feature_cols]
        y_train = train_data['us_aqi']
        
        logger.info(f"Training with {len(feature_cols)} features on {len(X_train)} samples")
        
        if is_incremental and existing_model is not None:
            # Incremental training: retrain existing model with new data
            logger.info("Performing incremental training on existing model")
            
            # Use same parameters as existing model but retrain
            # Note: LightGBM doesn't support true incremental learning
            # So we retrain with combined data approach or use existing model as init_model
            
            train_dataset = lgb.Dataset(X_train, label=y_train)
            
            # Train new model with existing model as starting point (warm start)
            model = lgb.train(
                params={
                    'objective': 'regression',
                    'metric': 'mae',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': 42
                },
                train_set=train_dataset,
                num_boost_round=100,  # Add fewer rounds for incremental
                init_model=existing_model,  # Use existing model as starting point
                callbacks=[lgb.early_stopping(stopping_rounds=10)]
            )
        else:
            # Fresh training
            logger.info("Performing fresh model training")
            
            train_dataset = lgb.Dataset(X_train, label=y_train)
            
            model = lgb.train(
                params={
                    'objective': 'regression',
                    'metric': 'mae',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': 42
                },
                train_set=train_dataset,
                num_boost_round=200,
                callbacks=[lgb.early_stopping(stopping_rounds=20)]
            )
        
        return model, feature_cols

    def evaluate_model(self, model, test_data, feature_cols):
        """Evaluate model performance"""
        if test_data is None or len(test_data) == 0:
            logger.warning("No test data available for evaluation")
            return None, None, None
        
        try:
            X_test = test_data[feature_cols]
            y_test = test_data['us_aqi']
            
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Model evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
            return mae, rmse, r2
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None, None, None

    def save_model_to_registry(self, project, model, all_features, training_info, metrics):
        """Save model to Hopsworks registry with training metadata"""
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
                description=f"LightGBM AQI forecaster - Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Trained on {training_info['total_samples']} samples"
            )
            
            # Save artifacts
            os.makedirs(self.config['ARTIFACT_DIR'], exist_ok=True)
            joblib.dump(model, os.path.join(self.config['ARTIFACT_DIR'], "lgb_model.pkl"))
            joblib.dump(all_features, os.path.join(self.config['ARTIFACT_DIR'], "lgb_features.pkl"))
            
            # Save training metadata
            joblib.dump(training_info, os.path.join(self.config['ARTIFACT_DIR'], "training_info.pkl"))
            
            model_meta.save(self.config['ARTIFACT_DIR'])
            logger.info("Model saved to registry successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model to registry: {e}")
            return False

    def run_pipeline(self):
        """Main training pipeline"""
        try:
            logger.info("Starting AQI training pipeline...")
            
            # Connect to Hopsworks
            project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name=self.config['FEATURE_GROUP_NAME'], version=self.config['FEATURE_GROUP_VER'])
            
            # Load ALL data from feature store
            df_raw = fg.read()
            df_raw = df_raw.sort_values("time", ascending=True).reset_index(drop=True)
            
            logger.info(f"Loaded {len(df_raw)} total rows from feature store")
            
            # Load existing model and training info
            existing_model, existing_features, training_info, model_version = self.load_existing_model_and_info(project)
            
            # Prepare training data
            result = self.prepare_training_data(df_raw, training_info, existing_model)
            if len(result) == 4:  # No new data case
                train_data, full_work_data, test_data, no_new_data = result
                latest_data_timestamp = None
            else:
                train_data, full_work_data, test_data, no_new_data, latest_data_timestamp = result
            
            if no_new_data:
                logger.info("No new data available for training")
                return {"status": "no_new_data", "total_samples_trained": training_info.get('total_samples', 0)}
            
            # Train model
            is_incremental = existing_model is not None
            model, all_features = self.train_model(train_data, existing_model, is_incremental)
            
            # Evaluate model
            mae, rmse, r2 = self.evaluate_model(model, test_data or train_data, all_features)
            metrics = {"mae": mae, "rmse": rmse, "r2": r2} if mae is not None else {}
            
            # Update training info
            new_samples_count = len(train_data)
            training_info.update({
                "last_training_time": datetime.now(timezone.utc),
                "total_samples": training_info.get('total_samples', 0) + new_samples_count,
                "last_data_timestamp": latest_data_timestamp.isoformat() if latest_data_timestamp else None
            })
            
            # Save to registry
            success = self.save_model_to_registry(project, model, all_features, training_info, metrics)
            
            result = {
                "status": "success",
                "training_type": "incremental" if is_incremental else "fresh",
                "new_samples": int(new_samples_count),
                "total_samples_trained": int(training_info['total_samples']),
                "metrics": metrics,
                "registry_saved": success,
                "last_data_timestamp": str(latest_data_timestamp)
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