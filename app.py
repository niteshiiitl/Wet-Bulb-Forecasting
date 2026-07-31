import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="AI Weather Forecaster", page_icon="🌦️", layout="wide")

st.title("🌦️ Next-Gen AI Weather Forecaster")
st.markdown("### 🚨 Predicting dangerous Wet-Bulb Temperature (WBT) levels 10 days in advance.")
st.markdown("---")

# --- 1. LOAD DATA ---
@st.cache_data
def load_dashboard_data():
    try:
        # Load the test inputs
        test_raw = pd.read_csv('test.csv')
        # Load your ALREADY COMPUTED predictions
        predictions = pd.read_csv('final_predictions.csv')
        return test_raw, predictions
    except FileNotFoundError:
        st.error("⚠️ Ensure 'test.csv' and 'final_predictions.csv' are uploaded to GitHub.")
        return None, None

test_data, preds_data = load_dashboard_data()

# --- 2. LOAD DAY 1 DEMO MODEL ---
@st.cache_resource
def load_demo_model():
    try:
        # Load ONLY the Day 1 LightGBM model for the live demo
        model = joblib.load('saved_model/lgb_day_1.pkl')
        return model
    except FileNotFoundError:
        st.error("⚠️ Ensure 'saved_model/lgb_day_1.pkl' is uploaded for the live demo.")
        return None

demo_model = load_demo_model()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Select Location")
if test_data is not None and preds_data is not None:
    selected_row = st.sidebar.selectbox("Choose a Test Coordinate (row_id):", test_data['row_id'].values)
    
    st.sidebar.markdown("---")
    st.sidebar.success("✅ Pre-Computed Data Active")
    if demo_model:
        st.sidebar.success("✅ Day-1 Live Inference Model Ready")

# --- MAIN DASHBOARD ---
if test_data is not None and preds_data is not None:
    
    # --- TAB LAYOUT ---
    tab1, tab2, tab3 = st.tabs(["10-Day Horizon (Batch)", "Live Inference Demo (Day 1)", "System Architecture"])
    
    with tab1:
        st.subheader("🚀 10-Day AI Forecast (Pre-computed Batch Inference)")
        st.markdown("This tab displays the output of the full 30-model ensemble running on GPU-accelerated backend hardware.")
        
        row_preds = preds_data[preds_data['row_id'] == selected_row]
        
        if not row_preds.empty:
            forecast_values = row_preds.drop(columns=['row_id']).values.flatten()
            
            m1, m2, m3 = st.columns(3)
            peak_wbt = max(forecast_values)
            m1.metric("Day 1 Forecast", f"{forecast_values[0]:.1f} °C")
            m2.metric("10-Day Peak", f"{peak_wbt:.1f} °C")
            
            if peak_wbt >= 32.0:
                m3.error("🔴 Extreme Heat Hazard")
            elif peak_wbt >= 28.0:
                m3.warning("🟠 High Risk (Caution)")
            else:
                m3.success("🟢 Safe / Normal")

            days = [f"Day {i}" for i in range(1, 11)]
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=days, y=forecast_values, 
                mode='lines+markers+text',
                text=[f"{v:.1f}°C" for v in forecast_values],
                textposition="top center",
                name='Predicted WBT',
                line=dict(color='#00E676', width=3),
                marker=dict(size=10, symbol='diamond')
            ))
            
            fig.add_hline(y=35, line_dash="solid", line_color="red", annotation_text="Lethal Limit (35°C)", annotation_position="top left")
            fig.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="Severe Danger Zone (30°C)", annotation_position="top left")
            
            fig.update_layout(
                title="Wet Bulb Temperature (WBT) Projection",
                xaxis_title="Forecast Horizon",
                yaxis_title="Temperature (°C)",
                template="plotly_dark",
                hovermode="x unified",
                yaxis=dict(range=[min(forecast_values) - 3, max(max(forecast_values) + 3, 37)]),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🔬 Live Inference Engine Demo")
        st.markdown("To demonstrate the pipeline's operational capability within Streamlit's 1GB memory limit, this tab executes a live inference pass using the Day-1 LightGBM base model.")
        
        display_data = test_data[test_data['row_id'] == selected_row]
        st.write("Input Telemetry:")
        st.dataframe(display_data.drop(columns=['row_id', 'date', 'location_id'], errors='ignore').T, use_container_width=True)
        
        if st.button("Run Day-1 Live Inference", type="primary"):
            if demo_model:
                with st.spinner("Executing live prediction..."):
                    # Note: You will need to ensure the features passed to X_live match what the model expects
                    feature_cols = [c for c in display_data.columns if c not in ['row_id', 'location_id', 'date', 'day_index', 'WBT', 'is_test', 'time_sort']]
                    X_live = display_data[feature_cols]
                    
                    try:
                        # Assuming the model predicts a delta that needs to be added to the baseline
                        baseline_wbt = display_data['WBT'].values[0] if 'WBT' in display_data.columns else 25.0
                        pred_delta = demo_model.predict(X_live)[0]
                        final_pred = baseline_wbt + pred_delta
                        
                        st.success("✅ Live Inference Successful!")
                        st.metric("Live Day 1 Prediction", f"{final_pred:.2f} °C")
                    except Exception as e:
                        st.error(f"Inference failed. Check feature alignment: {e}")
            else:
                 st.error("Live demo model not loaded.")

    with tab3:
        st.subheader("System Architecture & Infrastructure")
        st.markdown("""
        **Frontend / Dashboard (This App):**
        * Designed for high-speed, sub-second latency visualization.
        * Uses pre-computed telemetry for the full 10-day ensemble to avoid memory swap failures on the 1GB Streamlit Cloud tier.
        * Implements a lightweight live-inference demo to prove operational capability.
        
        **Backend / Batch Inference:**
        * Execution environment: GPU-accelerated instances (e.g., Google Colab T4).
        * Incorporates advanced thermodynamic feature engineering (e.g., Vapor Pressure Deficit).
        * Utilizes a Progressive Chain architecture across 30 models (LightGBM, XGBoost, CatBoost).
        """)
