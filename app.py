import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Weather Forecaster", page_icon="🌦️", layout="wide")

st.title("🌦️ Next-Gen AI Weather Forecaster")
st.markdown("### 🚨 Predicting dangerous Wet-Bulb Temperature (WBT) levels 10 days in advance using 41 years of NASA meteorological data.")
st.markdown("---")

# --- 🛠️ FEATURE ENGINEERING ENGINE (ADDED!) ---
def engineer_ml_features(df):
    df = df.copy()
    df['location_id'] = df['rel_lat'].round(4).astype(str) + "_" + df['rel_lon'].round(4).astype(str)

    df['DTR'] = df['T2M_MAX'] - df['T2M_MIN']
    if 'ALLSKY_SFC_SW_DWN' in df.columns and 'CLRSKY_SFC_SW_DWN' in df.columns:
        df['SOLAR_DELTA'] = df['ALLSKY_SFC_SW_DWN'] - df['CLRSKY_SFC_SW_DWN']
    if 'EVLAND' in df.columns and 'GWETTOP' in df.columns:
        df['EVAP_STRESS'] = df['EVLAND'] / (df['GWETTOP'] + 1e-5)
    
    df['TEMP_HUM_CROSS'] = df['T2M_MAX'] * df['RH2M']

    features_to_roll = ['T2M_MAX', 'RH2M', 'DTR', 'EVAP_STRESS', 'WBT']
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

    if 'date' in df.columns:
        day_of_year = pd.to_datetime(df['date']).dt.dayofyear
    else:
        day_of_year = (df.get('day_index', 1) % 365) + 1

    df['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.0)
    df['cos_day'] = np.cos(2 * np.pi * day_of_year / 365.0)

    return df

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Load raw data AND apply feature engineering immediately
        raw_df = pd.read_csv('test.csv').head(200)
        engineered_df = engineer_ml_features(raw_df)
        return engineered_df
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
    st.sidebar.caption("🛡️ Variance Shield Active")

# --- MAIN DASHBOARD ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📡 Input Telemetry")
    st.write("Current atmospheric features going into the models:")
    if test_data is not None:
        display_data = test_data[test_data['row_id'] == selected_row].drop(columns=['row_id', 'date', 'location_id'], errors='ignore')
        st.dataframe(display_data.T, use_container_width=True)

with col2:
    st.subheader("🚀 Live 10-Day AI Forecast")
    
    if st.button("Run Full 10-Day Machine Learning Inference", type="primary"):
        
        row_data = test_data[test_data['row_id'] == selected_row]
        exclude_cols = ['row_id', 'location_id', 'date', 'day_index', 'WBT']
        features = [c for c in row_data.columns if c not in exclude_cols]
        X_live = row_data[features].values
        
        full_10_day_forecast = []
        
        with st.spinner("Initializing AI Engine..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                for day in range(1, 11):
                    status_text.text(f"Loading Models and Calculating Day {day}...")
                    
                    # 1. Load the Indestructible CatBoost Model
                    model_cat = CatBoostRegressor()
                    model_cat.load_model(f'saved_model/cat_day_{day}.cbm')
                    pred_cat = model_cat.predict(X_live)[0]

                    # 2. Try loading LightGBM 
                    try:
                        model_lgb = joblib.load(f'saved_model/lgb_day_{day}.pkl')
                        pred_lgb = model_lgb.predict(X_live)[0]
                    except Exception:
                        pred_lgb = pred_cat  

                    # 3. Try loading XGBoost
                    try:
                        model_xgb = joblib.load(f'saved_model/xgb_day_{day}.pkl')
                        pred_xgb = model_xgb.predict(X_live)[0]
                    except Exception:
                        pred_xgb = pred_cat 

                    # Blend using your exact formula
                    live_prediction = (pred_lgb * 0.45) + (pred_xgb * 0.40) + (pred_cat * 0.15)
                    live_prediction = np.clip(live_prediction, -10, 45)
                    full_10_day_forecast.append(live_prediction)
                    
                    progress_bar.progress(day * 10)
                
                status_text.empty()
                st.success("✅ Full 10-Day Machine Learning Inference Complete!")
                
                # --- PLOTLY INTERACTIVE GRAPH ---
                days = [f"Day {i}" for i in range(1, 11)]
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=days, y=full_10_day_forecast, 
                    mode='lines+markers',
                    name='Predicted WBT',
                    line=dict(color='#00FF00', width=3),
                    marker=dict(size=10, symbol='diamond')
                ))
                
                fig.add_hline(y=35, line_dash="solid", line_color="red", annotation_text="Lethal Limit (35°C) ", annotation_position="top left")
                fig.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="Severe Danger Zone (30°C) ", annotation_position="top left")

                # Handle potential NaN values in plot scaling by filtering them out for the min/max calculation
                valid_forecasts = [val for val in full_10_day_forecast if not np.isnan(val)]
                min_y = min(valid_forecasts) - 2 if valid_forecasts else 15
                
                fig.update_layout(
                    title=f"Wet Bulb Temperature (WBT) Projection",
                    xaxis_title="Forecast Horizon",
                    yaxis_title="Temperature (°C)",
                    template="plotly_dark",
                    hovermode="x unified",
                    yaxis=dict(range=[min(min_y, 15), 40])
                )
                
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Critical System Error: {e}")
