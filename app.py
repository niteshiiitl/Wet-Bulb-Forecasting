import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Weather Forecaster", page_icon="🌦️", layout="wide")

st.title("🌦️ Next-Gen AI Weather Forecaster")
st.markdown("### 🚨 Predicting dangerous Wet-Bulb Temperature (WBT) levels 10 days in advance.")
st.markdown("---")

# --- 1. THERMODYNAMIC FEATURE ENGINEERING (For Live Demo) ---
def engineer_ml_features(df):
    df = df.copy()
    df['location_id'] = df['rel_lat'].round(4).astype(str) + "_" + df['rel_lon'].round(4).astype(str)
    df['DTR'] = df['T2M_MAX'] - df['T2M_MIN']
    
    if 'ALLSKY_SFC_SW_DWN' in df.columns and 'CLRSKY_SFC_SW_DWN' in df.columns:
        df['SOLAR_DELTA'] = df['ALLSKY_SFC_SW_DWN'] - df['CLRSKY_SFC_SW_DWN']
    if 'EVLAND' in df.columns and 'GWETTOP' in df.columns:
        df['EVAP_STRESS'] = df['EVLAND'] / (df['GWETTOP'] + 1e-5)
        
    # Vapor Pressure Deficit (VPD) via Tetens Equation
    if 'T2M_MAX' in df.columns and 'RH2M' in df.columns:
        df['E_SAT'] = 0.61078 * np.exp((17.27 * df['T2M_MAX']) / (df['T2M_MAX'] + 237.3))
        df['E_ACT'] = df['E_SAT'] * (df['RH2M'] / 100.0)
        df['VPD'] = df['E_SAT'] - df['E_ACT']
        
    df['TEMP_HUM_CROSS'] = df['T2M_MAX'] * df['RH2M']
    
    # Rolling Windows & Thermal Inertia
    features_to_roll = ['T2M_MAX', 'RH2M', 'DTR', 'VPD', 'EVAP_STRESS', 'WBT']
    grouped = df.groupby('location_id')
    for feat in features_to_roll:
        if feat in df.columns:
            df[f'{feat}_ewma_3'] = grouped[feat].transform(lambda x: x.ewm(span=3, adjust=False).mean())
            df[f'{feat}_ewma_14'] = grouped[feat].transform(lambda x: x.ewm(span=14, adjust=False).mean())
            df[f'{feat}_max_7d'] = grouped[feat].transform(lambda x: x.rolling(7, min_periods=1).max())
            df[f'{feat}_min_7d'] = grouped[feat].transform(lambda x: x.rolling(7, min_periods=1).min())
            df[f'{feat}_lag_1'] = grouped[feat].shift(1)
            df[f'{feat}_lag_2'] = grouped[feat].shift(2)
            df[f'{feat}_lag_3'] = grouped[feat].shift(3)
            df[f'{feat}_inertia'] = df[feat] - df[f'{feat}_lag_1']

    if 'date' in df.columns:
        day_of_year = pd.to_datetime(df['date']).dt.dayofyear
    else:
        day_of_year = (df.get('day_index', 1) % 365) + 1
        
    df['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.0)
    df['cos_day'] = np.cos(2 * np.pi * day_of_year / 365.0)
    
    return df

# --- 2. LOAD COMPRESSED DATA & ENGINEER FEATURES ---
@st.cache_data
def load_dashboard_data():
    try:
        # Load raw data and align them for the live inference demo
        context_raw = pd.read_csv('context.csv')
        test_raw = pd.read_csv('test.csv')
        
        context = engineer_ml_features(context_raw)
        test = engineer_ml_features(test_raw)
        context['is_test'], test['is_test'] = 0, 1
        
        if 'date' in context.columns:
            context['time_sort'] = pd.to_datetime(context['date']).astype(int)
        else:
            context['time_sort'] = context.get('day_index', 0)
        test['time_sort'] = test.get('day_index', 0) + 10000000000
        
        combined_test = pd.concat([context, test], ignore_index=True)
        combined_test = combined_test.sort_values(by=['location_id', 'time_sort']).reset_index(drop=True)
        
        exclude_cols = ['row_id', 'location_id', 'date', 'day_index', 'WBT', 'is_test', 'time_sort']
        features = [c for c in combined_test.columns if c not in exclude_cols and not c.startswith('target_day_')]
        
        # Fill NA values for continuity
        for col in features:
            if combined_test[col].isnull().any():
                combined_test[col] = combined_test.groupby('location_id')[col].transform(lambda x: x.ffill().bfill())
        combined_test['WBT'] = combined_test.groupby('location_id')['WBT'].transform(lambda x: x.ffill())
        
        test_ready = combined_test[combined_test['is_test'] == 1].copy()

        # Load the COMPRESSED predictions file for instant UI rendering
        predictions = pd.read_csv('final_predictions.csv.gz', compression='gzip')
        
        return test_ready, predictions, features
    except FileNotFoundError:
        st.error("⚠️ Error: Ensure 'context.csv', 'test.csv', and 'final_predictions.csv.gz' are in the repository.")
        return None, None, None

test_data, preds_data, feature_cols = load_dashboard_data()

# --- 3. LOAD DAY 1 DEMO MODEL ---
@st.cache_resource
def load_demo_model():
    try:
        # Load ONLY the Day 1 LightGBM model for evaluation purposes
        model = joblib.load('saved_model/lgb_day_1.pkl')
        return model
    except FileNotFoundError:
        return None

demo_model = load_demo_model()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Select Location")
if test_data is not None and preds_data is not None:
    selected_row = st.sidebar.selectbox("Choose a Test Coordinate (row_id):", test_data['row_id'].values)
    
    st.sidebar.markdown("---")
    st.sidebar.success("✅ GZIP Pre-Computed Data Active")
    if demo_model:
        st.sidebar.success("✅ Live Inference Engine Ready")
    else:
        st.sidebar.warning("⚠️ Live Demo Model Not Found")

# --- MAIN DASHBOARD ---
if test_data is not None and preds_data is not None:
    
    tab1, tab2, tab3 = st.tabs(["10-Day Horizon (Batch)", "Live Inference Demo (Day 1)", "System Architecture"])
    
    # --- TAB 1: INSTANT BATCH RENDERING ---
    with tab1:
        st.subheader("🚀 10-Day AI Forecast (GPU Batch Output)")
        st.markdown("Visualizing the output of the 30-model ensemble running on off-site GPU-accelerated hardware.")
        
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
            elif peak_wbt >= 24.0:
                m3.info("🟡 Moderate Caution")
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
            
    # --- TAB 2: LIVE DAY-1 INFERENCE DEMO ---
    with tab2:
        st.subheader("🔬 Live Inference Engine Demo")
        st.markdown("Demonstrates operational capability within Streamlit's 1GB memory limit. Executes a live predictive pass using the LightGBM Day-1 base model.")
        
        display_data = test_data[test_data['row_id'] == selected_row]
        st.write("Current Thermodynamic Features (Input Tensor):")
        st.dataframe(display_data.drop(columns=['row_id', 'date', 'location_id', 'is_test', 'time_sort'], errors='ignore').T, use_container_width=True)
        
        if st.button("Initialize & Run Day-1 Inference", type="primary"):
            if demo_model:
                with st.spinner("Executing live target delta transformation..."):
                    X_live = display_data[feature_cols]
                    
                    try:
                        baseline_wbt = display_data['WBT'].values[0]
                        pred_delta = demo_model.predict(X_live)[0]
                        final_pred = np.clip(baseline_wbt + pred_delta, -10, 45)
                        
                        st.success("✅ Live Inference Computed Successfully!")
                        st.metric("Live Calculated Day 1 Prediction", f"{final_pred:.2f} °C", delta=f"{pred_delta:.2f} °C Shift")
                    except Exception as e:
                        st.error(f"Inference failed. Check feature space alignment: {e}")
            else:
                 st.error("Live demo model not loaded. Please upload 'saved_model/lgb_day_1.pkl'.")

    # --- TAB 3: SYSTEM ARCHITECTURE ---
    with tab3:
        st.subheader("⚙️ System Architecture & Engineering Trade-offs")
        st.markdown("""
        **Frontend Analytics Layer (Streamlit Cloud):**
        * Engineered for high-speed, sub-second latency visualization.
        * Compresses massive CSV data outputs into heavily optimized `.gz` archives to bypass standard GitHub repository limits.
        * Decouples the 30-model ensemble execution from the UI thread to guarantee 100% uptime within a constrained 1GB RAM environment.
        
        **Backend Compute Layer (Google Colab T4 GPU):**
        * Executes the computationally expensive Progressive Chain architecture.
        * Performs target residual ($\Delta$) transformations based on baseline shifts.
        * Calculates real-world atmospheric physics mathematically, applying the Tetens Equation to determine localized Vapor Pressure Deficit (VPD).
        """)
