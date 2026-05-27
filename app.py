import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
import plotly.graph_objects as go
import os

st.set_page_config(page_title="AI Weather Forecaster", page_icon="🌦️", layout="wide")

st.title("🌦️ Next-Gen AI Weather Forecaster")
st.markdown("### 🚨 Predicting dangerous Wet-Bulb Temperature (WBT) levels 10 days in advance using 41 years of NASA meteorological data.")
st.markdown("---")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Load the test dataset (we only need a few rows for the demo to save memory)
        df = pd.read_csv('test.csv').head(200)
        return df
    except FileNotFoundError:
        st.error("⚠️ test.csv not found! Make sure it is in the same folder as app.py")
        return None

test_data = load_data()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Forecast Parameters")
if test_data is not None:
    selected_row = st.sidebar.selectbox("Select a Location/Timepoint to Forecast:", test_data['row_id'].values)
    
    st.sidebar.markdown("---")
    st.sidebar.success("✅ Live Inference Engine Active")
    st.sidebar.info("🤖 Models: XGBoost + LightGBM + CatBoost")

# --- MAIN DASHBOARD ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📡 Input Telemetry")
    st.write("Current atmospheric features going into the models:")
    if test_data is not None:
        # Show the raw data the model is about to look at
        display_data = test_data[test_data['row_id'] == selected_row].drop(columns=['row_id', 'date', 'location_id'], errors='ignore')
        st.dataframe(display_data.T, use_container_width=True)

with col2:
    st.subheader("🚀 Live 10-Day AI Forecast")
    
    if st.button("Run Full 10-Day Machine Learning Inference", type="primary"):
        
        # Grab the exact row of data the user selected
        row_data = test_data[test_data['row_id'] == selected_row]
        
        # Drop the non-numeric columns
        exclude_cols = ['row_id', 'location_id', 'date', 'day_index', 'WBT']
        features = [c for c in row_data.columns if c not in exclude_cols]
        X_live = row_data[features].values
        
        full_10_day_forecast = []
        
        with st.spinner("Initializing AI Engine..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Loop through all 10 days
                for day in range(1, 11):
                    status_text.text(f"Loading Models and Calculating Day {day}...")
                    
                    # Load the models for this specific day
                    model_lgb = joblib.load(f'saved_models/lgb_day_{day}.pkl')
                    model_xgb = joblib.load(f'saved_models/xgb_day_{day}.pkl')
                    
                    model_cat = CatBoostRegressor()
                    model_cat.load_model(f'saved_models/cat_day_{day}.cbm')

                    # Predict
                    pred_lgb = model_lgb.predict(X_live)[0]
                    pred_xgb = model_xgb.predict(X_live)[0]
                    pred_cat = model_cat.predict(X_live)[0]

                    # Blend using your exact formula
                    live_prediction = (pred_lgb * 0.45) + (pred_xgb * 0.40) + (pred_cat * 0.15)
                    
                    # Apply your exact clipping shield
                    live_prediction = np.clip(live_prediction, -10, 45)
                    
                    full_10_day_forecast.append(live_prediction)
                    
                    # Update progress bar
                    progress_bar.progress(day * 10)
                
                status_text.empty()
                st.success("✅ Full 10-Day Machine Learning Inference Complete!")
                
                # --- PLOTLY INTERACTIVE GRAPH ---
                days = [f"Day {i}" for i in range(1, 11)]
                
                fig = go.Figure()
                
                # Main Prediction Line
                fig.add_trace(go.Scatter(
                    x=days, y=full_10_day_forecast, 
                    mode='lines+markers',
                    name='Predicted WBT',
                    line=dict(color='#00FF00', width=3),
                    marker=dict(size=10, symbol='diamond')
                ))
                
                # Add Danger Threshold Lines for the Presentation Pitch
                fig.add_hline(y=35, line_dash="solid", line_color="red", annotation_text="Lethal Limit (35°C) ", annotation_position="top left")
                fig.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="Severe Danger Zone (30°C) ", annotation_position="top left")

                fig.update_layout(
                    title=f"Wet Bulb Temperature (WBT) Projection for {selected_row}",
                    xaxis_title="Forecast Horizon",
                    yaxis_title="Temperature (°C)",
                    template="plotly_dark",
                    hovermode="x unified",
                    yaxis=dict(range=[min(min(full_10_day_forecast)-2, 15), 40]) # Lock Y-axis scale to look good
                )
                
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error loading models. Did you put the 'saved_models' folder in the same directory? Error: {e}")