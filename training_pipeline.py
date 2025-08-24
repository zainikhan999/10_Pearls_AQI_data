"""
LightGBM AQI Model Retraining Script
===================================
This script retrains the LightGBM AQI forecasting model incrementally.
It checks for new data since the last training and updates the model accordingly.

Author: Your Name
Date: 2025
"""

# Import required libraries
import os
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta

import hopsworks
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hsml.model import ModelSchema, Schema

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Hopsworks feature store configuration
FEATURE_GROUP_NAME = "aqi_weather_features"
FEATURE_GROUP_VER = 2
API_KEY = os.environ["HOPSWORKS_API_KEY"]
# Location configuration (Rawalpindi, Pakistan)
LATITUDE = 33.5973
LONGITUDE = 73.0479
TZ = "Asia/Karachi"

# Model configuration
HORIZON_H = 72  # Forecast horizon in hours
MAX_LAG_H = 120  # Maximum lag for features

# Directory structure
ARTIFACT_DIR = "lgb_aqi_artifacts"
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Model artifact paths
MODEL_PATH = os.path.join(ARTIFACT_DIR, "lgb_model.pkl")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "lgb_features.pkl")
TIMESTAMP_PATH = os.path.join(ARTIFACT_DIR, "last_trained_timestamp.pkl")

# =============================================================================
# HELPER FUNCTIONS
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


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression performance metrics.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted values
    
    Returns:
        tuple: (MAE, RMSE, R²) scores
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def load_existing_artifacts():
    """
    Load existing model artifacts if they exist.
    
    Returns:
        tuple: (model, features, last_timestamp)
    """
    model = None
    features = None
    last_timestamp = None
    
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading existing model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    
    if os.path.exists(FEATURES_PATH):
        print(f"[INFO] Loading feature list from: {FEATURES_PATH}")
        features = joblib.load(FEATURES_PATH)
    
    if os.path.exists(TIMESTAMP_PATH):
        print(f"[INFO] Loading last training timestamp from: {TIMESTAMP_PATH}")
        last_timestamp = joblib.load(TIMESTAMP_PATH)
        print(f"[INFO] Last training timestamp: {last_timestamp}")
    
    return model, features, last_timestamp


def save_artifacts(model, features, last_timestamp):
    """
    Save model artifacts to disk.
    
    Args:
        model: Trained LightGBM model
        features (list): List of feature column names
        last_timestamp: Last training timestamp
    """
    joblib.dump(model, MODEL_PATH)
    joblib.dump(features, FEATURES_PATH)
    joblib.dump(last_timestamp, TIMESTAMP_PATH)
    
    print(f"[INFO] Artifacts saved to: {ARTIFACT_DIR}")
    print(f"[INFO] Last training timestamp: {last_timestamp}")


def create_evaluation_plots(y_true, y_pred, r2_score):
    """
    Create evaluation plots for model performance.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted values
        r2_score (float): R-squared score
    """
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
    
    print(f"[INFO] Evaluation plot saved to: {plot_path}")
    return plot_path


# =============================================================================
# MAIN RETRAINING FUNCTION
# =============================================================================

def retrain_model():
    """
    Main function to retrain the LightGBM AQI model.
    """
    print("=" * 80)
    print("🚀 LIGHTGBM AQI MODEL RETRAINING")
    print("=" * 80)
    
    # Step 1: Connect to Hopsworks
    print("\n[1/8] 🔌 Connecting to Hopsworks...")
    try:
        project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VER)
        print("✅ Successfully connected to Hopsworks")
    except Exception as e:
        print(f"❌ Failed to connect to Hopsworks: {e}")
        return
    
    # Step 2: Load existing artifacts
    print("\n[2/8] 📦 Loading existing model artifacts...")
    existing_model, existing_features, last_trained_timestamp = load_existing_artifacts()
    
    # Step 3: Fetch data from feature store
    print("\n[3/8] 📊 Fetching data from feature store...")
    try:
        df_raw = fg.read()
        df_raw = df_raw.sort_values("time", ascending=True).reset_index(drop=True)
        print(f"✅ Fetched {len(df_raw)} records from feature store")
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        return
    
    # Validate required columns
    required_cols = ["time", "pm_10", "pm_25", "carbon_monoxidegm", 
                    "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]
    missing_cols = [col for col in required_cols if col not in df_raw.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return
    
    # Prepare data
    df = df_raw[required_cols].copy()
    df["time_utc"] = ensure_utc(df["time"])
    
    # Step 4: Determine training data range
    print("\n[4/8] 📅 Determining training data range...")
    
    if last_trained_timestamp is not None:
        last_trained_utc = ensure_utc(pd.Series([last_trained_timestamp]))[0]
        new_data = df[df["time_utc"] > last_trained_utc].copy()
        
        if len(new_data) == 0:
            print("ℹ️  No new data available since last training. Exiting.")
            return
        
        print(f"✅ Found {len(new_data)} new records since {last_trained_timestamp}")
        print(f"📈 New data range: {new_data['time_utc'].min()} to {new_data['time_utc'].max()}")
        
        # Include context for lag features (30 days of historical data)
        context_cutoff = last_trained_utc - pd.Timedelta(days=30)
        training_data = df[df["time_utc"] >= context_cutoff].copy()
    else:
        print("ℹ️  No previous training timestamp found. Training from scratch.")
        training_data = df.copy()
    
    print(f"📊 Training dataset size: {len(training_data)} records")
    print(f"📅 Training data range: {training_data['time_utc'].min()} to {training_data['time_utc'].max()}")
    
    # Step 5: Feature engineering
    print("\n[5/8] ⚙️  Creating features...")
    
    base_features = ["pm_10", "pm_25", "carbon_monoxidegm", 
                    "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
    
    # Create lag features
    work_df = training_data.set_index("time_utc")[base_features + ["us_aqi"]].copy()
    work_df = create_lag_features(work_df, base_features)
    work_df.dropna(inplace=True)
    
    if len(work_df) == 0:
        print("❌ No valid data after feature engineering. Exiting.")
        return
    
    # Get feature columns
    all_feature_cols = [col for col in work_df.columns if col != "us_aqi"]
    
    # Ensure feature consistency with existing model
    if existing_features is not None:
        if set(all_feature_cols) != set(existing_features):
            print("⚠️  Feature mismatch detected!")
            print(f"   Existing features: {len(existing_features)}")
            print(f"   New features: {len(all_feature_cols)}")
            
            missing_features = set(existing_features) - set(all_feature_cols)
            if missing_features:
                print(f"❌ Missing required features: {missing_features}")
                return
            
            print("🔄 Using existing feature set for consistency")
            all_feature_cols = existing_features
    
    # Prepare X and y
    X = work_df[all_feature_cols]
    y = work_df["us_aqi"]
    
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"🎯 Target vector shape: {y.shape}")
    
    # Step 6: Train/validation split
    print("\n[6/8] ✂️  Creating train/validation split...")
    
    split_idx = int(len(work_df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"🔥 Training set: {len(X_train)} samples")
    print(f"🔍 Validation set: {len(X_val)} samples")
    
    # Step 7: Model training
    print("\n[7/8] 🤖 Training LightGBM model...")
    
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
        'force_col_wise': True  # Better for many features
    }
    
    # Train the model
    print("🔄 Starting model training...")
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
    
    print("✅ Model training completed!")
    
    # Step 8: Model evaluation
    print("\n[8/8] 📊 Evaluating model performance...")
    
    # Make predictions
    y_pred_val = model.predict(X_val)
    mae, rmse, r2 = calculate_metrics(y_val, y_pred_val)
    
    # Display results
    print(f"📈 VALIDATION METRICS:")
    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R²:   {r2:.4f}")
    
    # Create evaluation plots
    create_evaluation_plots(y_val, y_pred_val, r2)
    
    # Save artifacts
    new_last_timestamp = work_df.index.max()
    save_artifacts(model, all_feature_cols, new_last_timestamp)
    
    # Save to Hopsworks Model Registry with versioning
    print("\n🏪 Saving model to Hopsworks Model Registry...")
    try:
        # Create schema
        input_schema = Schema(X_train)
        output_schema = Schema(y_train)
        model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
        
        # Get model registry
        mr = project.get_model_registry()
        
        # Determine next version
        model_name = "lgb_aqi_forecaster"
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            existing_models = mr.get_models(model_name)
            if existing_models:
                latest_version = max([model.version for model in existing_models])
                next_version = latest_version + 1
                print(f"📊 Found existing model versions. Next version: {next_version}")
            else:
                next_version = 1
                print(f"🆕 First model version: {next_version}")
        except:
            next_version = 1
            print(f"🆕 Creating new model. Version: {next_version}")
        
        # Create and save model
        model_meta = mr.python.create_model(
            name=model_name,
            version=next_version,
            metrics={
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "retrain_timestamp": current_time
            },
            model_schema=model_schema,
            description=f"LightGBM AQI forecaster v{next_version} - Retrained on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with {len(X_train)} samples"
        )
        
        model_meta.save(ARTIFACT_DIR)
        print(f"✅ Model v{next_version} successfully saved to registry!")
        
        # Save version info locally
        version_info = {
            "model_name": model_name,
            "version": next_version,
            "retrain_timestamp": current_time,
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "training_samples": len(X_train),
            "last_data_timestamp": str(new_last_timestamp)
        }
        
        version_path = os.path.join(ARTIFACT_DIR, "model_version_info.json")
        with open(version_path, 'w') as f:
            json.dump(version_info, f, indent=2)
        print(f"📝 Version info saved to: {version_path}")
        
    except Exception as e:
        print(f"⚠️  Failed to save to Model Registry: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 MODEL RETRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📁 Artifacts directory: {os.path.abspath(ARTIFACT_DIR)}")
    print(f"📊 Model performance:")
    print(f"   • MAE:  {mae:.2f}")
    print(f"   • RMSE: {rmse:.2f}")
    print(f"   • R²:   {r2:.4f}")
    print(f"⏰ Training data up to: {new_last_timestamp}")
    
    # Display version info if available
    try:
        version_path = os.path.join(ARTIFACT_DIR, "model_version_info.json")
        if os.path.exists(version_path):
            with open(version_path, 'r') as f:
                version_info = json.load(f)
            print(f"🔢 Model version: {version_info['version']}")
            print(f"📝 Model name: {version_info['model_name']}")
    except:
        pass
    
    print("=" * 80)


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Run the retraining script.
    
    Usage:
        python retrain_lgb_aqi_model.py
    """
    try:
        retrain_model()
    except KeyboardInterrupt:
        print("\n⚠️  Script interrupted by user")
    except Exception as e:
        print(f"\n❌ Script failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Script execution finished")