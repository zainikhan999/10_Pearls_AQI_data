# run_inference_push_fg.py
"""
Run LSTM inference (UTC-safe), validate, and push forecasts to Hopsworks Feature Groups.
- Appends history to FG:  aqi_forecasts (schema: datetime_utc, predicted_us_aqi, model_version, run_id, created_at, git_commit)
- Overwrites/inserts into FG: aqi_forecasts_latest (same schema) for fast reads by dashboards
Notes:
 - HOPSWORKS_API_KEY must be set in env.
 - If a FG does not exist it will attempt to create one (basic schema inference).
"""

import os
import uuid
import argparse
from datetime import datetime
import hopsworks
import joblib
import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import load_model
API_KEY = os.environ["HOPSWORKS_API_KEY"]

# ---- Config (tweak if needed) ----
PROJECT_NAME = "weather_aqi"
MODEL_NAME   = "aqi_lstm_forecaster"
MODEL_VER    = "latest"   # "latest" or integer string
FG_FEATURES_NAME = "aqi_weather_features"  # input features FG name
FG_FEATURES_VERSION = 2
TIMESTEPS    = 50
FORECAST_H   = 72
LAT, LON     = 33.5973, 73.0479
PKT_TZ       = "Asia/Karachi"

# Output FG names
FG_HISTORY_NAME = "aqi_forecasts"
FG_HISTORY_VERSION = 1
FG_LATEST_NAME = "aqi_forecasts_latest"
FG_LATEST_VERSION = 1

# Validation rules
EXPECTED_ROWS = FORECAST_H  # expected forecast rows

def read_future_pollutants(lat=LAT, lon=LON, pak_tz=PKT_TZ, days=4, hours_needed=FORECAST_H):
    air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "forecast_days": days,
        "timezone": pak_tz,
    }
    resp = requests.get(air_url, params=params)
    resp.raise_for_status()
    payload = resp.json()
    if "hourly" not in payload:
        raise RuntimeError(f"Open-Meteo response missing 'hourly': {payload}")
    df_future = pd.DataFrame({
        "time": payload["hourly"]["time"],
        "pm_10": payload["hourly"]["pm10"],
        "pm_25": payload["hourly"]["pm2_5"],
        "carbon_monoxidegm": payload["hourly"]["carbon_monoxide"],
        "nitrogen_dioxide": payload["hourly"]["nitrogen_dioxide"],
        "sulphur_dioxide": payload["hourly"]["sulphur_dioxide"],
        "ozone": payload["hourly"]["ozone"],
    })
    # parse and localize to PKT (explicit)
    df_future["time"] = pd.to_datetime(df_future["time"])
    # If API returned tz-aware strings this will preserve them; otherwise localize
    try:
        # if naive, localize
        if df_future["time"].dt.tz is None:
            df_future["time"] = df_future["time"].dt.tz_localize(pak_tz, nonexistent="shift_forward", ambiguous="NaT")
    except Exception:
        df_future["time"] = df_future["time"].dt.tz_localize(pak_tz, nonexistent="shift_forward", ambiguous="NaT")

    # convert to UTC for alignment with history FG (which we read as UTC)
    df_future_utc = df_future.copy()
    df_future_utc["time"] = df_future_utc["time"].dt.tz_convert("UTC")
    df_future_utc = df_future_utc.sort_values("time").reset_index(drop=True).iloc[:hours_needed]
    return df_future_utc

def ensure_fg_exists(fs, name, version, df_sample):
    """Try to get FG; if not present, create a simple FG with inferred schema.
       This uses fs.create_feature_group if available. Behavior may vary by Hopsworks version."""
    try:
        fg = fs.get_feature_group(name=name, version=version)
        return fg
    except Exception as e:
        print(f"⚠️ Feature group {name} v{version} not found: {e}. Attempting to create it.")
        # Build schema from df_sample
        # Map pandas dtypes to simple types expected by Hopsworks
        from hopsworks.featurestore import FeatureGroup, Feature
        schema = []
        for col, dtype in df_sample.dtypes.items():
            # choose simple type hints; Hopsworks will infer more accurately when using the SDK
            if pd.api.types.is_integer_dtype(dtype):
                t = "int"
            elif pd.api.types.is_float_dtype(dtype):
                t = "float"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                t = "timestamp"
            else:
                t = "string"
            schema.append(Feature(name=col, feature_type=t))
        # Create FG with minimal settings; primary_key could be ['datetime_utc','run_id'] but we'll leave none
        try:
            fg = fs.create_feature_group(name=name,
                                         version=version,
                                         primary_key=[],
                                         features=schema,
                                         description=f"Auto-created FG {name} by inference script")
            print(f"Created FG {name} v{version}")
            return fg
        except Exception as e2:
            print(f"Could not automatically create FG {name}: {e2}")
            raise

def push_df_to_fg(fs, name, version, df_to_insert):
    """Insert DataFrame into a FG (create FG if missing)."""
    fg = None
    try:
        fg = fs.get_feature_group(name=name, version=version)
    except Exception:
        fg = ensure_fg_exists(fs, name, version, df_to_insert)

    # insert
    try:
        fg.insert(df_to_insert, write_options={"wait_for_job": False})
    except Exception as e:
        print(f"Error inserting into FG {name}: {e}")
        # try blocking insert for better error visibility
        fg.insert(df_to_insert, write_options={"wait_for_job": True})
    print(f"Inserted {len(df_to_insert)} rows to FG {name} v{version}")

def main(args):
    if not API_KEY:
        raise EnvironmentError("HOPSWORKS_API_KEY not set in env")

    # Connect and load model artifacts
    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # load model (allow "latest")
    model_meta = mr.get_model(MODEL_NAME, version=MODEL_VER)
    model_dir = model_meta.download()
    model = load_model(os.path.join(model_dir, "lstm_aqi_model.h5"))
    X_scaler = joblib.load(os.path.join(model_dir, "X_scaler.pkl"))
    y_scaler = joblib.load(os.path.join(model_dir, "y_scaler.pkl"))
    features = joblib.load(os.path.join(model_dir, "features.pkl"))  # list of feature column names

    # feature store: read historical FG (input features)
    fg = fs.get_feature_group(name=FG_FEATURES_NAME, version=FG_FEATURES_VERSION)
    hist_df = fg.read()
    # ensure time column is parsed as UTC tz-aware
    hist_df["time"] = pd.to_datetime(hist_df["time"], utc=True)
    hist_df = hist_df.sort_values("time")

    if len(hist_df) < TIMESTEPS:
        raise ValueError(f"Need at least {TIMESTEPS} historical rows in FG {FG_FEATURES_NAME} v{FG_FEATURES_VERSION}. Found {len(hist_df)}")

    # pick last TIMESTEPS rows (aligned by time)
    last_hist = hist_df.set_index("time").iloc[-TIMESTEPS:]
    last_hist_X = last_hist[features].astype(float).copy()

    # fetch future pollutants and align to UTC
    future_df = read_future_pollutants(lat=args.lat, lon=args.lon, pak_tz=PKT_TZ, days=4, hours_needed=FORECAST_H)

    # validate feature presence in future payload
    for f in features:
        if f not in future_df.columns:
            raise RuntimeError(f"Feature '{f}' not present in future pollutant payload. features: {features}")

    # scale
    X_last_scaled = X_scaler.transform(last_hist_X.values)
    X_future_scaled = X_scaler.transform(future_df[features].astype(float).values)

    # rolling forecast
    input_seq = X_last_scaled.copy()
    predicted = []
    for i in range(FORECAST_H):
        pred_scaled = model.predict(input_seq.reshape(1, TIMESTEPS, len(features)), verbose=0)
        pred_val = y_scaler.inverse_transform(pred_scaled)[0][0]
        predicted.append(float(pred_val))
        next_x = X_future_scaled[i]
        input_seq = np.vstack([input_seq[1:], next_x])

    # build time axes: base off last_hist max timestamp (UTC)
    last_hist_ts_utc = last_hist.index.max()
    future_times_utc = pd.date_range(
        start=last_hist_ts_utc + pd.Timedelta(hours=1),
        periods=FORECAST_H,
        freq="H",
        tz="UTC"
    )
    future_times_pkt = future_times_utc.tz_convert(PKT_TZ)

    # prepare forecast DataFrame
    forecast_df = pd.DataFrame({
        "datetime_utc": future_times_utc.tz_convert("UTC").tz_localize(None),  # naive UTC (Hopsworks friendly)
        "predicted_us_aqi": predicted,
    })
    # add local PKT column for convenience (string)
    forecast_df["datetime_pkt"] = future_times_pkt.astype(str)

    # metadata / provenance
    run_id = str(uuid.uuid4())
    model_version = str(model_meta.version) if hasattr(model_meta, "version") else str(MODEL_VER)
    created_at = datetime.utcnow()
    git_commit = os.environ.get("GIT_COMMIT", None)  # pass from CI via env

    # add metadata columns
    forecast_df["model_version"] = model_version
    forecast_df["run_id"] = run_id
    forecast_df["created_at"] = created_at
    forecast_df["git_commit"] = git_commit

    # basic validation
    if len(forecast_df) != EXPECTED_ROWS:
        raise RuntimeError(f"Validation failed: expected {EXPECTED_ROWS} rows, got {len(forecast_df)}")
    if forecast_df["predicted_us_aqi"].isnull().any():
        raise RuntimeError("Validation failed: found null predictions")

    # Save local copy
    out_csv = args.out or "72hr_forecast_utc_aligned.csv"
    forecast_df.to_csv(out_csv, index=False)
    print(f"✅ Saved local CSV: {out_csv}")

    # Prepare DF for FG insertion: choose columns matching desired FG schema
    df_to_insert = forecast_df[["datetime_utc", "predicted_us_aqi", "model_version", "run_id", "created_at", "git_commit"]].copy()

    # Push to history FG (append)
    print("📥 Pushing forecast to Hopsworks history FG...")
    push_df_to_fg(fs, FG_HISTORY_NAME, FG_HISTORY_VERSION, df_to_insert)

    # Also push to 'latest' FG for fast reads by dashboard.
    # Strategy: write the same rows to a separate FG named `aqi_forecasts_latest`.
    print("📥 Pushing forecast to Hopsworks latest-FG (for fast reads)...")
    push_df_to_fg(fs, FG_LATEST_NAME, FG_LATEST_VERSION, df_to_insert)

    print(f"✅ Done. run_id={run_id}, model_version={model_version}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None, help="Local CSV output path")
    parser.add_argument("--model-version", type=str, default=MODEL_VER, help="Model version (or 'latest')")
    parser.add_argument("--lat", type=float, default=LAT)
    parser.add_argument("--lon", type=float, default=LON)
    args = parser.parse_args()
    main(args)
