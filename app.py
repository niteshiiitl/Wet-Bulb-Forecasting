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

# --- 🛠️ FEATURE ENGINEERING ENGINE (ALIGNED WITH V3 TRAINING) ---
def engineer_ml_features(df):
    df = df.copy()
    df['location_id'] = df['rel_lat'].round(4).astype(str) + "_" + df['rel_lon'].round(4).astype(str)

    # Core Thermodynamics
    df['DTR'] = df['T2M_MAX'] - df['T2M_MIN']
    if 'ALLSKY_SFC_SW_DWN' in df.columns and 'CLRSKY_SFC_SW_DWN' in df.columns:
        df['SOLAR_DELTA'] = df['ALLSKY_SFC_SW_DWN'] - df['CLRSKY_SFC_SW_DWN']
    if 'EVLAND' in df.columns and 'GWETTOP' in df.columns:
        df['EVAP_STRESS'] = df['EVLAND'] / (df['GWETTOP'] + 1e-5)
    
    df['TEMP_HUM_CROSS'] = df['T2M_MAX'] * df['RH2M']

    # Decaying Memory and Rolling Stats
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

    # Cyclical Seasons
    if 'date' in df.columns:
        day_of_year = pd.to_datetime(df['date']).dt.dayofyear
    else:
        day_of_year = (df.get('day_index', 1) % 365) + 1

    df['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.0)
    df['cos_day'] = np.cos(2 * np.pi * day_of_year / 365.0)

    return df

# --- LOAD DATA (WITH CONTEXT ALIGNMENT TO PREVENT NANS) ---
@st.cache_data
def load_and_prepare_data():
    try:
        # Load raw data
        context_raw = pd.read_csv('context.csv')
        test_raw = pd.read_csv('test.csv')

        # Apply feature engineering
        context = engineer_ml_features(context_raw)
        test = engineer_ml_features(test_raw)

        # Alignment Setup (Mirroring the Jupyter Notebook)
        context['is_test'], test['is_test'] = 0, 1
        
        if 'date' in context.columns:
            context['time_sort'] = pd.to_datetime(context['date']).astype(int)
        else:
            context['time_sort'] = context.get('day_index', 0)
            
        test['time_sort'] = test.get('day_index', 0) + 10000000000

        # Combine, sort, and reset to compute valid continuous lags
        combined_test = pd.concat([context, test], ignore_index=True)
        combined_test = combined_test.sort_values(by=['location_id', 'time_sort']).reset_index(drop=True)

        # Identify model features
        exclude_cols = ['row_id', 'location_id', 'date', 'day_index', 'WBT', 'is_test', 'time_sort']
        features = [c for c in combined_test.columns if c not in exclude_cols and not c.startswith('target_day_')]

        # Forward fill missing test features and the WBT baseline 
        for col in features:
            if combined_test[col].isnull().any():
                combined_test[col] = combined_test.groupby('location_id')[col].transform(lambda x: x.ffill().bfill())

        combined_test['WBT'] = combined_test.groupby('location_id')['WBT'].transform(lambda x: x.ffill())

        # Extract only the test rows for UI
        test_ready = combined_test[combined_test['is_test'] == 1].copy()
        
        return test_ready, features
    except FileNotFoundError as e:
        st.error(f"⚠️ Missing Data File: Ensure 'test.csv' and 'context.csv' are in the same folder as app.py.")
        return None, None

test_data, feature_cols = load_and_prepare_data()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Forecast Parameters")
if test_data is not None:
    selected_row = st.sidebar.selectbox("Select a Location/Timepoint to Forecast:", test_data['row_id'].values)
    
    st.sidebar.markdown("---")
    st.sidebar.success("✅ Live Inference Engine Active")
    st.sidebar.info("🤖 Models: XGBoost + LightGBM + CatBoost")
    st.sidebar.caption("🛡️ Variance Shield Active")

# --- MAIN DASHBOARD ---
if test_data is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📡 Input Telemetry")
        st.write("Current atmospheric features going into the models:")
        display_data = test_data[test_data['row_id'] == selected_row].drop(columns=['row_id', 'date', 'location_id', 'is_test', 'time_sort'], errors='ignore')
        st.dataframe(display_data.T, use_container_width=True)

    with col2:
        st.subheader("🚀 Live 10-Day AI Forecast")
        
        if st.button("Run Full 10-Day Machine Learning Inference", type="primary"):
            
            row_data = test_data[test_data['row_id'] == selected_row]
            
            # CRITICAL FIX 1: Pass as a raw NumPy array to match the (60,) shape models trained on
            X_live = row_data[feature_cols].values 
            
            # CRITICAL FIX 2: Establish the baseline temperature for Delta Inference
            baseline_wbt = row_data['WBT'].values[0] 
            
            full_10_day_forecast = []
            
            with st.spinner("Initializing AI Engine..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    for day in range(1, 11):
                        status_text.text(f"Loading Models and Calculating Day {day} Delta...")
                        
                        # Apply V3 Horizon-Specific Blending Weights
                        if (day - 1) < 4:
                            w_lgb, w_xgb, w_cat = 0.50, 0.40, 0.10  # Short-Term Trust
                        else:
                            w_lgb, w_xgb, w_cat = 0.20, 0.30, 0.50  # Long-Term Caution
                        
                        # 1. Load the CatBoost Model
                        try:
                            model_cat = CatBoostRegressor()
                            model_cat.load_model(f'saved_model/cat_day_{day}.cbm')
                            pred_cat_delta = model_cat.predict(X_live)[0]
                        except Exception as e:
                            st.warning(f"CatBoost Day {day} failed: {e}")
                            pred_cat_delta = np.nan

                        # 2. Load LightGBM 
                        try:
                            model_lgb = joblib.load(f'saved_model/lgb_day_{day}.pkl')
                            pred_lgb_delta = model_lgb.predict(X_live)[0]
                        except Exception:
                            pred_lgb_delta = pred_cat_delta if not np.isnan(pred_cat_delta) else np.nan

                        # 3. Load XGBoost
                        try:
                            model_xgb = joblib.load(f'saved_model/xgb_day_{day}.pkl')
                            pred_xgb_delta = model_xgb.predict(X_live)[0]
                        except Exception:
                            pred_xgb_delta = pred_cat_delta if not np.isnan(pred_cat_delta) else np.nan

                        # Blend Delta
                        valid_preds = [p for p in [pred_lgb_delta, pred_xgb_delta, pred_cat_delta] if not np.isnan(p)]
                        
                        if len(valid_preds) == 3:
                            blended_delta = (pred_lgb_delta * w_lgb) + (pred_xgb_delta * w_xgb) + (pred_cat_delta * w_cat)
                        elif len(valid_preds) > 0:
                            blended_delta = np.mean(valid_preds) 
                        else:
                            blended_delta = np.nan 

                        # Convert Delta back to Absolute Temperature
                        if not np.isnan(blended_delta):
                            absolute_prediction = baseline_wbt + blended_delta
                            absolute_prediction = np.clip(absolute_prediction, -10, 45)
                        else:
                            absolute_prediction = np.nan
                            
                        full_10_day_forecast.append(absolute_prediction)
                        
                        progress_bar.progress(day * 10)
                    
                    status_text.empty()
                    st.success("✅ Full 10-Day Horizon Inference Complete!")
                    
                    # --- PLOTLY INTERACTIVE GRAPH ---
                    days = [f"Day {i}" for i in range(1, 11)]
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=days, y=full_10_day_forecast, 
                        mode='lines+markers',
                        name='Predicted WBT',
                        line=dict(color='#00FF00', width=3),
                        marker=dict(size=10, symbol='diamond'),
                        connectgaps=True
                    ))
                    
                    fig.add_hline(y=35, line_dash="solid", line_color="red", annotation_text="Lethal Limit (35°C) ", annotation_position="top left")
                    fig.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="Severe Danger Zone (30°C) ", annotation_position="top left")

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
                    st.error(f"Critical Inference Error: {e}")
