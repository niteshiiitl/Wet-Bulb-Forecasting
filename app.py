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

# --- 🛠️ FEATURE ENGINEERING ENGINE (FULL V3 ALIGNMENT) ---
def engineer_ml_features(df):
    df = df.copy()
    df['location_id'] = df['rel_lat'].round(4).astype(str) + "_" + df['rel_lon'].round(4).astype(str)

    # 1. Core Thermodynamics
    df['DTR'] = df['T2M_MAX'] - df['T2M_MIN']
    if 'ALLSKY_SFC_SW_DWN' in df.columns and 'CLRSKY_SFC_SW_DWN' in df.columns:
        df['SOLAR_DELTA'] = df['ALLSKY_SFC_SW_DWN'] - df['CLRSKY_SFC_SW_DWN']
    if 'EVLAND' in df.columns and 'GWETTOP' in df.columns:
        df['EVAP_STRESS'] = df['EVLAND'] / (df['GWETTOP'] + 1e-5)
    
    # 2. Advanced Physics: Vapor Pressure Deficit (VPD)
    if 'T2M_MAX' in df.columns and 'RH2M' in df.columns:
        df['E_SAT'] = 0.61078 * np.exp((17.27 * df['T2M_MAX']) / (df['T2M_MAX'] + 237.3))
        df['E_ACT'] = df['E_SAT'] * (df['RH2M'] / 100.0)
        df['VPD'] = df['E_SAT'] - df['E_ACT']
    
    df['TEMP_HUM_CROSS'] = df['T2M_MAX'] * df['RH2M']

    # 3. Decaying Memory, Rolling Stats & Thermal Inertia
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
            # Thermal Inertia
            df[f'{feat}_inertia'] = df[feat] - df[f'{feat}_lag_1']

    # 4. Cyclical Seasons
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

        # Fill missing features dynamically
        for col in features:
            if combined_test[col].isnull().any():
                combined_test[col] = combined_test.groupby('location_id')[col].transform(lambda x: x.ffill().bfill())

        combined_test['WBT'] = combined_test.groupby('location_id')['WBT'].transform(lambda x: x.ffill())

        test_ready = combined_test[combined_test['is_test'] == 1].copy()
        
        return test_ready, features
    except FileNotFoundError:
        st.error("⚠️ Missing Data File: Ensure 'test.csv' and 'context.csv' are in the same directory as app.py.")
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
            
            # Pass as a DataFrame with feature columns to preserve feature names for tree models
            X_live = row_data[feature_cols]
            baseline_wbt = row_data['WBT'].values[0]
            
            full_10_day_forecast = []
            
            with st.spinner("Initializing AI Engine..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    for day in range(1, 11):
                        status_text.text(f"Running Inference for Day {day}...")
                        
                        # Horizon-Specific Weighting
                        if (day - 1) < 4:
                            w_lgb, w_xgb, w_cat = 0.50, 0.40, 0.10
                        else:
                            w_lgb, w_xgb, w_cat = 0.20, 0.30, 0.50
                        
                        pred_cat_delta, pred_lgb_delta, pred_xgb_delta = np.nan, np.nan, np.nan
                        
                        # 1. CatBoost
                        try:
                            model_cat = CatBoostRegressor()
                            model_cat.load_model(f'saved_model/cat_day_{day}.cbm')
                            pred_cat_delta = model_cat.predict(X_live)[0]
                        except Exception:
                            pass

                        # 2. LightGBM 
                        try:
                            model_lgb = joblib.load(f'saved_model/lgb_day_{day}.pkl')
                            pred_lgb_delta = model_lgb.predict(X_live)[0]
                        except Exception:
                            pass

                        # 3. XGBoost
                        try:
                            model_xgb = joblib.load(f'saved_model/xgb_day_{day}.pkl')
                            pred_xgb_delta = model_xgb.predict(X_live)[0]
                        except Exception:
                            pass

                        # Dynamic Weighted Blend (handles missing models gracefully)
                        valid_models = []
                        weights = []
                        
                        if not np.isnan(pred_lgb_delta):
                            valid_models.append(pred_lgb_delta)
                            weights.append(w_lgb)
                        if not np.isnan(pred_xgb_delta):
                            valid_models.append(pred_xgb_delta)
                            weights.append(w_xgb)
                        if not np.isnan(pred_cat_delta):
                            valid_models.append(pred_cat_delta)
                            weights.append(w_cat)
                        
                        if valid_models:
                            norm_weights = np.array(weights) / sum(weights)
                            blended_delta = np.sum(np.array(valid_models) * norm_weights)
                        else:
                            blended_delta = 0.0  # Fallback: no temperature change

                        # Reverse Delta Transformation
                        absolute_prediction = baseline_wbt + blended_delta
                        absolute_prediction = np.clip(absolute_prediction, -10, 45)
                        
                        full_10_day_forecast.append(absolute_prediction)
                        progress_bar.progress(day * 10)
                    
                    status_text.empty()
                    st.success("✅ Full 10-Day Horizon Inference Complete!")
                    
                    # --- PLOTLY INTERACTIVE GRAPH ---
                    days = [f"Day {i}" for i in range(1, 11)]
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=days, y=full_10_day_forecast, 
                        mode='lines+markers+text',
                        text=[f"{v:.1f}°C" for v in full_10_day_forecast],
                        textposition="top center",
                        name='Predicted WBT',
                        line=dict(color='#00E676', width=3),
                        marker=dict(size=10, symbol='diamond')
                    ))
                    
                    fig.add_hline(y=35, line_dash="solid", line_color="red", annotation_text="Lethal Limit (35°C)", annotation_position="top left")
                    fig.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="Severe Danger Zone (30°C)", annotation_position="top left")
                    
                    fig.update_layout(
                        title="Wet Bulb Temperature (WBT) 10-Day Projection",
                        xaxis_title="Forecast Horizon",
                        yaxis_title="Temperature (°C)",
                        template="plotly_dark",
                        hovermode="x unified",
                        yaxis=dict(range=[min(full_10_day_forecast) - 3, max(max(full_10_day_forecast) + 3, 37)])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Critical Inference Error: {e}")
