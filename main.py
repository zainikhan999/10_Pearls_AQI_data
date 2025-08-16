import streamlit as st
import pandas as pd
import hopsworks
import os
API_KEY = os.environ["HOPSWORKS_API_KEY"]

# === Connect to Hopsworks ===
project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
fs = project.get_feature_store()

# Load the predictions feature group
fg = fs.get_feature_group(name="aqi_predictions", version=1)

# Read as dataframe
df = fg.read()

# Ensure datetime is parsed
df['forecast_date'] = pd.to_datetime(df['forecast_date'])
df['prediction_time'] = pd.to_datetime(df['prediction_time'])

# === Keep only the latest prediction run ===
latest_run_time = df['prediction_time'].max()
latest_preds = df[df['prediction_time'] == latest_run_time]

# Sort by forecast_date
latest_preds = latest_preds.sort_values("forecast_date")

# === Streamlit UI ===
st.title("🌍 AQI Forecast Dashboard")
st.write("Showing the latest forecast from model")

# Show latest prediction date/time
st.info(f"Latest prediction run: {latest_run_time}")

# Display table
st.dataframe(latest_preds[['forecast_date', 'us_aqi']])

# Optional: Plot
st.line_chart(latest_preds.set_index("forecast_date")["us_aqi"])
