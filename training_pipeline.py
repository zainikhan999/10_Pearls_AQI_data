"""
LightGBM AQI Model Retraining Script - Enhanced with Data Accumulation
====================================================================
This script retrains the LightGBM AQI forecasting model incrementally.
It properly accumulates historical data instead of losing it with each run.

Key improvements:
1. Accumulates historical data in feature store instead of discarding it
2. Fetches only new data since last update to avoid API limits
3. Maintains consistent feature engineering across runs
4. Better data management and error handling
5. Fixed datetime comparison and feature group read issues
"""

import os
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta, timezone

import hopsworks
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hsml.model import ModelSchema, Schema

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Hopsworks feature store configuration
FEATURE_GROUP_NAME = "aqi_weather_features"
FEATURE_GROUP_VER = 2
HISTORICAL_DATA_FG = "aqi_historical_accumulator"  # New: accumulated historical data
HISTORICAL_DATA_VER = 1
API_KEY = os.environ["HOPSWORKS_API_KEY"]

# Location configuration (Rawalpindi, Pakistan)
LATITUDE = 33.5973
LONGITUDE = 73.0479
TZ = "Asia/Karachi"

# Model configuration
HORIZON_H = 72  # Forecast horizon in hours
MAX_LAG_H = 120  # Maximum lag for features
HISTORICAL_DAYS = 90  # Keep 90 days of accumulated data

# Directory structure (mainly for plots and temporary storage)
ARTIFACT_DIR = "lgb_aqi_artifacts"
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Local backup paths (optional - Hopsworks is primary storage)
MODEL_PATH = os.path.join(ARTIFACT_DIR, "lgb_model_backup.pkl")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "lgb_features_backup.pkl")
TIMESTAMP_PATH = os.path.join(ARTIFACT_DIR, "last_trained_timestamp_backup.pkl")

# Base features for consistency
BASE_FEATURES = ["pm_10", "pm_25", "carbon_monoxidegm", 
                "nitrogen_dioxide", "sulphur_dioxide", "ozone"]

# =============================================================================
# DATA ACCUMULATION FUNCTIONS
# =============================================================================

def fetch_pollutant_data_api(start_hours_ago, end_hours_ago=0):
    """Fetch pollutant data from API for a specific time range."""
    print(f"Fetching API data from {start_hours_ago}h to {end_hours_ago}h ago...")
    
    try:
        now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_time = (now_utc - timedelta(hours=start_hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_time = (now_utc - timedelta(hours=end_hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Fetch both pollutants and AQI
        air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
            "start": start_time,
            "end": end_time,
            "timezone": "UTC",
        }
        
        response = requests.get(air_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame({
            "time": pd.to_datetime(data["hourly"]["time"]),
            "pm_10": data["hourly"]["pm10"],
            "pm_25": data["hourly"]["pm2_5"],
            "carbon_monoxidegm": data["hourly"]["carbon_monoxide"],
            "nitrogen_dioxide": data["hourly"]["nitrogen_dioxide"],
            "sulphur_dioxide": data["hourly"]["sulphur_dioxide"],
            "ozone": data["hourly"]["ozone"],
            "us_aqi": data["hourly"]["us_aqi"]
        })
        
        # Remove any rows where AQI is null (can't train without target)
        df = df.dropna(subset=['us_aqi'])
        
        print(f"Fetched {len(df)} valid records from API")
        return df
        
    except Exception as e:
        print(f"Failed to fetch from API: {e}")
        return pd.DataFrame()


def get_or_create_historical_fg(fs):
    """Get existing historical data feature group or create new one."""
    try:
        historical_fg = fs.get_feature_group(
            name=HISTORICAL_DATA_FG, 
            version=HISTORICAL_DATA_VER
        )
        print(f"Found existing historical data feature group: {HISTORICAL_DATA_FG}")
        return historical_fg
        
    except Exception as e:
        print(f"Could not get existing feature group: {e}")
        print(f"Creating new historical data feature group: {HISTORICAL_DATA_FG}")
        
        # Create feature group for accumulated historical data
        historical_fg = fs.create_feature_group(
            name=HISTORICAL_DATA_FG,
            version=HISTORICAL_DATA_VER,
            description="Accumulated historical pollutant and AQI data for model training",
            primary_key=["time"],
            event_time="time",
            online_enabled=False,
            statistics_config=False
        )
        return historical_fg


def safe_read_feature_group(feature_group):
    """Safely read from feature group with proper error handling."""
    try:
        # Try different read methods
        data = None
        
        # Method 1: Direct read
        try:
            data = feature_group.read()
            if data is not None and len(data) > 0:
                print(f"Successfully read {len(data)} records using direct read")
                return data
        except Exception as e:
            print(f"Direct read failed: {e}")
        
        # Method 2: Read with query
        try:
            query = feature_group.select_all()
            data = query.read()
            if data is not None and len(data) > 0:
                print(f"Successfully read {len(data)} records using query")
                return data
        except Exception as e:
            print(f"Query read failed: {e}")
        
        # Method 3: Read with limits
        try:
            data = feature_group.read(limit=10000)
            if data is not None and len(data) > 0:
                print(f"Successfully read {len(data)} records using limited read")
                return data
        except Exception as e:
            print(f"Limited read failed: {e}")
        
        print("All read methods failed or returned empty data")
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Failed to read from feature group: {e}")
        return pd.DataFrame()


def update_accumulated_historical_data(fs):
    """Update the accumulated historical data with latest values."""
    print("Updating accumulated historical data...")
    
    try:
        historical_fg = get_or_create_historical_fg(fs)
        
        # Try to read existing data with safe method
        existing_data = safe_read_feature_group(historical_fg)
        
        if len(existing_data) > 0:
            # Ensure time column is datetime and UTC-aware
            existing_data['time'] = ensure_utc(existing_data['time'])
            last_time = existing_data['time'].max()
            print(f"Last data timestamp in store: {last_time}")
            
            # Calculate hours since last update
            now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            
            # Now we can safely compare, because last_time is already UTC
            hours_to_fetch = max(1, int((now_utc - last_time).total_seconds() / 3600))
            print(f"Need to fetch {hours_to_fetch} hours of new data")
            
            # Fetch only new data
            new_data = fetch_pollutant_data_api(hours_to_fetch, 0)
            
            if len(new_data) > 0:
                # Combine and deduplicate
                all_data = pd.concat([existing_data, new_data], ignore_index=True)
                all_data = all_data.drop_duplicates(subset=['time']).sort_values('time')
                print(f"Combined data: {len(all_data)} total records")
            else:
                print("No new data to add")
                all_data = existing_data
        else:
            print("No existing data found, fetching initial dataset")
            # First time - get last 30 days
            all_data = fetch_pollutant_data_api(24 * 30, 0)
        
        if len(all_data) == 0:
            raise ValueError("No data available for training")
        
        # Keep only recent data (last N days)
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=HISTORICAL_DAYS)
        
        # Corrected: Ensure all_data['time'] is timezone-aware before comparison
        all_data['time'] = ensure_utc(all_data['time'])
        
        # Convert cutoff_time to pandas timestamp for comparison
        cutoff_timestamp = pd.Timestamp(cutoff_time)
        
        recent_data = all_data[all_data['time'] >= cutoff_timestamp].copy()
        
        # Ensure proper data types
        for col in BASE_FEATURES + ['us_aqi']:
            if col in recent_data.columns:
                recent_data[col] = pd.to_numeric(recent_data[col], errors='coerce')
        
        # Remove any remaining null values
        recent_data = recent_data.dropna(subset=BASE_FEATURES + ['us_aqi'])
        
        # ... (rest of the function remains the same)
        print(f"Saving {len(recent_data)} records to historical feature group")
        print(f"Data range: {recent_data['time'].min()} to {recent_data['time'].max()}")
        
        try:
            historical_fg.delete_all()
            print("Cleared existing data from feature group")
        except Exception as e:
            print(f"Could not clear existing data: {e}")
        
        historical_fg.insert(recent_data, write_options={"wait_for_job": True})
        print("Successfully saved data to feature store")
        
        return recent_data
        
    except Exception as e:
        print(f"Failed to update historical data: {e}")
        import traceback
        traceback.print_exc()
        raise

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
        output_df[f"{feature}_roll_mean_24"] = output_df[feature].rolling(24, min_periods=12).mean()
        output_df[f"{feature}_roll_std_24"] = output_df[feature].rolling(24, min_periods=12).std()
        output_df[f"{feature}_roll_mean_72"] = output_df[feature].rolling(72, min_periods=24).mean()
        output_df[f"{feature}_roll_std_72"] = output_df[feature].rolling(72, min_periods=24).std()
    
    return output_df


def ensure_utc(timestamp_series):
    """Ensure timestamp series is in UTC timezone."""
    if isinstance(timestamp_series, (list, tuple)):
        timestamp_series = pd.Series(timestamp_series)
    
    ts = pd.to_datetime(timestamp_series)
    
    # Handle single timestamp vs series
    if hasattr(ts, 'dt'):
        if ts.dt.tz is None:
            return ts.dt.tz_localize("UTC")
        else:
            return ts.dt.tz_convert("UTC")
    else:
        # Single timestamp
        if ts.tz is None:
            return ts.tz_localize("UTC")
        else:
            return ts.tz_convert("UTC")


def calculate_metrics(y_true, y_pred):
    """Calculate regression performance metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def load_existing_model_from_registry(project):
    """Load existing model and metadata from Hopsworks Model Registry."""
    model = None
    features = None
    last_timestamp = None
    model_metrics = None
    
    try:
        # Get model registry
        mr = project.get_model_registry()
        model_name = "lgb_aqi_forecaster"
        
        # Get latest model version
        try:
            existing_models = mr.get_models(model_name)
            if existing_models:
                latest_model = max(existing_models, key=lambda x: x.version)
                print(f"Found existing model: {model_name} v{latest_model.version}")
                
                # Download model artifacts
                model_dir = latest_model.download()
                print(f"Downloaded model to: {model_dir}")
                
                # Load the actual model
                model_file = os.path.join(model_dir, "lgb_model.pkl")
                if os.path.exists(model_file):
                    model = joblib.load(model_file)
                    print("Successfully loaded LightGBM model")
                
                # Load features
                features_file = os.path.join(model_dir, "lgb_features.pkl")
                if os.path.exists(features_file):
                    features = joblib.load(features_file)
                    print(f"Loaded {len(features)} features from registry")
                
                # Load last timestamp
                timestamp_file = os.path.join(model_dir, "last_trained_timestamp.pkl")
                if os.path.exists(timestamp_file):
                    last_timestamp = joblib.load(timestamp_file)
                    print(f"Last training timestamp: {last_timestamp}")
                
                # Get model metrics
                model_metrics = latest_model.training_metrics
                if model_metrics:
                    print(f"Previous model metrics: R²={model_metrics.get('r2', 'N/A'):.4f}, RMSE={model_metrics.get('rmse', 'N/A'):.2f}")
                
            else:
                print(f"No existing models found for {model_name}")
                
        except Exception as e:
            print(f"No existing models found or error accessing registry: {e}")
            
    except Exception as e:
        print(f"Failed to access model registry: {e}")
    
    return model, features, last_timestamp, model_metrics


def save_artifacts(model, features, last_timestamp):
    """Save model artifacts to disk."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(features, FEATURES_PATH)
    joblib.dump(last_timestamp, TIMESTAMP_PATH)
    
    print(f"Artifacts saved to: {ARTIFACT_DIR}")
    print(f"Last training timestamp: {last_timestamp}")


def create_evaluation_plots(y_true, y_pred, r2_score):
    """Create evaluation plots for model performance."""
    plt.figure(figsize=(12, 8))
    
    # Scatter plot: Predicted vs Actual
    plt.subplot(2, 1, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual US AQI')
    plt.ylabel('Predicted US AQI')
    plt.title(f'Predicted vs Actual US AQI (R² = {r2_score:.4f})')
    plt.grid(True, alpha=0.3)
    
    # Residual plot
    plt.subplot(2, 1, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='red', linestyle='--', lw=2)
    plt.xlabel('Predicted US AQI')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with timestamp
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(PLOTS_DIR, f"model_evaluation_{timestamp_str}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation plot saved to: {plot_path}")
    return plot_path


# =============================================================================
# MAIN RETRAINING FUNCTION
# =============================================================================

def retrain_model():
    """Main function to retrain the LightGBM AQI model with accumulated data."""
    print("=" * 80)
    print("LIGHTGBM AQI MODEL RETRAINING - Enhanced with Data Accumulation")
    print("=" * 80)
    
    # Step 1: Connect to Hopsworks
    print("\n[1/8] Connecting to Hopsworks...")
    try:
        project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
        fs = project.get_feature_store()
        print("Successfully connected to Hopsworks")
    except Exception as e:
        print(f"Failed to connect to Hopsworks: {e}")
        return
    
    # Step 2: Load existing model from Hopsworks Model Registry
    print("\n[2/8] Loading existing model from Hopsworks Model Registry...")
    existing_model, existing_features, last_trained_timestamp, prev_metrics = load_existing_model_from_registry(project)
    
    # Step 3: Update accumulated historical data
    print("\n[3/8] Updating accumulated historical data...")
    try:
        training_data = update_accumulated_historical_data(fs)
        print(f"Training data ready: {len(training_data)} records")
        print(f"Data range: {training_data['time'].min()} to {training_data['time'].max()}")
    except Exception as e:
        print(f"Failed to update historical data: {e}")
        return
    
    # Step 4: Check if we have enough new data to warrant retraining
    print("\n[4/8] Checking if retraining is needed...")
    
    if last_trained_timestamp is not None:
        try:
            last_trained_utc = ensure_utc(pd.Series([last_trained_timestamp]))[0]
            training_data_time = pd.to_datetime(training_data['time'])
            new_data = training_data[training_data_time > last_trained_utc]
            
            if len(new_data) < 24:  # Less than 1 day of new data
                print(f"Only {len(new_data)} new records since last training. Skipping retraining.")
                return
            
            print(f"Found {len(new_data)} new records since {last_trained_timestamp}")
        except Exception as e:
            print(f"Error checking new data: {e}")
            print("Proceeding with full retraining")
    else:
        print("No previous training found. Training from scratch.")
    
    # Step 5: Feature engineering with consistent features
    print("\n[5/8] Creating features...")
    
    # Sort by time and set as index for feature engineering
    work_df = training_data.sort_values('time').set_index('time')
    
    # Create lag features
    work_df = create_lag_features(work_df, BASE_FEATURES)
    
    # Remove rows with insufficient data for lag features
    work_df = work_df.dropna(subset=[f"{feat}_lag_1" for feat in BASE_FEATURES])
    
    if len(work_df) == 0:
        print("No valid data after feature engineering. Exiting.")
        return
    
    # Get all feature columns (everything except target)
    all_feature_cols = [col for col in work_df.columns if col != "us_aqi"]
    
    # Enforce feature consistency with existing model
    if existing_features is not None:
        print(f"Enforcing feature consistency with existing model")
        print(f"Required features: {len(existing_features)}")
        
        # Ensure all required features exist
        missing_features = set(existing_features) - set(all_feature_cols)
        if missing_features:
            print(f"WARNING: Missing required features: {missing_features}")
            # Fill missing features with zeros
            for feature in missing_features:
                work_df[feature] = 0.0
        
        # Use exact same feature set
        all_feature_cols = existing_features
        print("Using identical feature set for consistency")
    else:
        print(f"Creating new feature set with {len(all_feature_cols)} features")
    
    # Prepare X and y
    X = work_df[all_feature_cols]
    y = work_df["us_aqi"]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Step 6: Train/validation split (time-based)
    print("\n[6/8] Creating train/validation split...")
    
    # Use last 20% of data for validation (time-based split)
    split_idx = int(len(work_df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Step 7: Model training
    print("\n[7/8] Training LightGBM model...")
    
    # Train or continue training the model
    model = train_or_continue_model(existing_model, X_train, y_train, X_val, y_val)
    
def train_or_continue_model(existing_model, X_train, y_train, X_val, y_val):
    """Train a new model or continue training an existing one."""
    
    # Prepare LightGBM datasets
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    # Model parameters
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
        'seed': 42,
        'force_col_wise': True
    }
    
    if existing_model is not None:
        print("Continuing training from existing model...")
        # Continue training from existing model
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=200,  # Fewer rounds for incremental training
            valid_sets=[lgb_val],
            init_model=existing_model,  # Use existing model as starting point
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(50)
            ]
        )
    else:
        print("Training new model from scratch...")
        # Train new model
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=800,
            valid_sets=[lgb_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(100)
            ]
        )
    
    return model
    
    print("Model training completed!")
    
    # Step 8: Model evaluation
    print("\n[8/8] Evaluating model performance...")
    
    # Make predictions
    y_pred_val = model.predict(X_val)
    mae, rmse, r2 = calculate_metrics(y_val, y_pred_val)
    
    # Display results
    print(f"VALIDATION METRICS:")
    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R²:   {r2:.4f}")
    
    # Create evaluation plots
    create_evaluation_plots(y_val, y_pred_val, r2)
    
    # Compare with previous model if available
    if prev_metrics:
        prev_r2 = prev_metrics.get('r2', 0)
        prev_rmse = prev_metrics.get('rmse', float('inf'))
        
        print(f"PERFORMANCE COMPARISON:")
        print(f"   Previous R²:  {prev_r2:.4f} → Current R²:  {r2:.4f} ({'↑' if r2 > prev_r2 else '↓'} {abs(r2 - prev_r2):.4f})")
        print(f"   Previous RMSE: {prev_rmse:.2f} → Current RMSE: {rmse:.2f} ({'↓' if rmse < prev_rmse else '↑'} {abs(rmse - prev_rmse):.2f})")
    
    # Save artifacts locally for backup (optional)
    new_last_timestamp = work_df.index.max()
    save_artifacts(model, all_feature_cols, new_last_timestamp)
    
    # Save to Hopsworks Model Registry
    print("\nSaving model to Hopsworks Model Registry...")
    try:
        # Create schema
        input_schema = Schema(X_train)
        output_schema = Schema(y_train)
        model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
        
        # Get model registry
        mr = project.get_model_registry()
        
        # Determine next version
        model_name = "lgb_aqi_forecaster"
        current_time = datetime.now()
        
        try:
            existing_models = mr.get_models(model_name)
            if existing_models:
                next_version = max([model.version for model in existing_models]) + 1
            else:
                next_version = 1
        except:
            next_version = 1
        
        # Create metrics dictionary
        metrics_dict = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "training_samples": int(len(X_train)),
            "validation_samples": int(len(X_val)),
            "retrain_timestamp_epoch": float(current_time.timestamp()),
            "feature_count": int(len(all_feature_cols))
        }
        
        # Create and save model
        model_meta = mr.python.create_model(
            name=model_name,
            version=next_version,
            metrics=metrics_dict,
            model_schema=model_schema,
            description=f"LightGBM AQI forecaster v{next_version} - Retrained with accumulated data on {current_time.strftime('%Y-%m-%d %H:%M:%S')} with {len(X_train)} samples. R²={r2:.4f}, RMSE={rmse:.2f}"
        )
        
        model_meta.save(ARTIFACT_DIR)
        print(f"Model v{next_version} successfully saved to registry!")
        
        # Save version info
        version_info = {
            "model_name": model_name,
            "version": next_version,
            "retrain_timestamp": current_time.strftime('%Y%m%d_%H%M%S'),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "feature_count": len(all_feature_cols),
            "last_data_timestamp": str(new_last_timestamp)
        }
        
        with open(os.path.join(ARTIFACT_DIR, "model_version_info.json"), 'w') as f:
            json.dump(version_info, f, indent=2)
        
    except Exception as e:
        print(f"Failed to save to Model Registry: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 80)
    print("MODEL RETRAINING COMPLETED WITH ACCUMULATED DATA!")
    print("=" * 80)
    print(f"Training data: {len(training_data)} total records accumulated")
    print(f"Model performance:")
    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R²:   {r2:.4f}")
    print(f"Training data up to: {new_last_timestamp}")
    print(f"Features: {len(all_feature_cols)}")
    print("=" * 80)


if __name__ == "__main__":
    """Run the enhanced retraining script."""
    try:
        retrain_model()
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
    except Exception as e:
        print(f"\nScript failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nScript execution finished")