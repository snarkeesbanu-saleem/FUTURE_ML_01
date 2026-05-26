import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Store Sales Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CACHING ======================
@st.cache_data(ttl=3600)
def load_data():
    base_path = "store-sales-time-series-forecasting (2)"
    try:
        train = pd.read_csv(f"{base_path}/train.csv", parse_dates=['date'])
        stores = pd.read_csv(f"{base_path}/stores.csv")
        oil = pd.read_csv(f"{base_path}/oil.csv", parse_dates=['date'])
        holidays = pd.read_csv(f"{base_path}/holidays_events.csv", parse_dates=['date'])
    except:
        # Fallback if folder structure is different
        train = pd.read_csv("train.csv", parse_dates=['date'])
        stores = pd.read_csv("stores.csv")
        oil = pd.read_csv("oil.csv", parse_dates=['date'])
        holidays = pd.read_csv("holidays_events.csv", parse_dates=['date'])

    train = train.merge(stores, on='store_nbr', how='left')
    oil['dcoilwtico'] = oil['dcoilwtico'].ffill()
    train = train.merge(oil, on='date', how='left')

    holiday_dates = set(holidays['date'].dt.date)
    train['is_holiday'] = train['date'].dt.date.isin(holiday_dates).astype(int)

    train['dayofweek'] = train['date'].dt.dayofweek
    train['month'] = train['date'].dt.month
    train['year'] = train['date'].dt.year
    train.rename(columns={'store_nbr': 'store_id'}, inplace=True)
    
    return train

@st.cache_resource
def load_xgboost_model():
    try:
        model = joblib.load("xgboost_model.pkl")
        return model
    except Exception as e:
        st.warning("XGBoost model not found. Using fallback forecast.")
        return None

@st.cache_resource
def load_label_encoder():
    try:
        return joblib.load("label_encoder.pkl")
    except:
        return None

# ====================== XGBoost FORECAST ======================
def xgboost_forecast(model, le, df, horizon, store_id, family):
    if model is None or le is None:
        # Fallback seasonal forecast
        last_week = df['sales'].tail(7).values
        forecast = [float(last_week[i % len(last_week)]) for i in range(horizon)]
        dates = [df['date'].max() + timedelta(days=i+1) for i in range(horizon)]
        return dates, forecast

    last_row = df.iloc[-1]
    last_sales = float(last_row['sales'])
    last_date = df['date'].max()

    try:
        family_encoded = int(le.transform([family])[0])
    except:
        family_encoded = 0

    future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
    predictions = []

    for future_date in future_dates:
        features = np.array([[
            int(store_id),
            family_encoded,
            int(last_row.get('onpromotion', 0)),
            float(last_row.get('dcoilwtico', 0)),
            int(last_row.get('is_holiday', 0)),
            future_date.weekday(),
            future_date.month,
            future_date.year,
            last_sales,           # lag_1
            last_sales            # lag_7 approx
        ]])
        
        pred = model.predict(features)[0]
        predictions.append(max(0.0, float(pred)))

    return future_dates, predictions

# ====================== MAIN ======================
def main():
    st.title("📊 Store Sales & Demand Forecasting Dashboard")
    st.caption("**FUTURE_ML_01** | XGBoost Powered")

    df = load_data()
    model = load_xgboost_model()
    le = load_label_encoder()

    # Sidebar
    st.sidebar.header("🔍 Controls")
    with st.sidebar.form("forecast_form"):
        store = st.selectbox("Select Store", sorted(df["store_id"].unique()))
        family = st.selectbox("Product Family", sorted(df["family"].unique()))
        horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
        submitted = st.form_submit_button("🚀 Generate Forecast")

    if not submitted:
        st.info("Select options in sidebar and click **Generate Forecast**")
        st.stop()

    filtered_df = df[(df['store_id'] == store) & (df['family'] == family)].copy()

    if filtered_df.empty:
        st.error("No data available for this selection.")
        st.stop()

    # Generate Forecast
    forecast_dates, forecast_values = xgboost_forecast(model, le, filtered_df, horizon, store, family)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", f"${filtered_df['sales'].sum():,.0f}")
    c2.metric("Avg Daily Sales", f"${filtered_df['sales'].mean():,.1f}")
    c3.metric("Forecast Days", f"{horizon}")
    c4.metric("Model Used", "XGBoost" if model is not None else "Seasonal")

    # Forecast Chart
    st.subheader(f"Forecast: Store {store} - {family}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['sales'], 
                           name="Actual Sales", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, 
                           name="XGBoost Forecast", line=dict(color="#ff7f0e", dash="dash"), mode="lines+markers"))
    fig.update_layout(height=550, template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    if model is not None:
        st.subheader("Feature Importance")
        feature_names = ['store_id', 'family', 'onpromotion', 'oil_price', 'holiday', 
                        'dayofweek', 'month', 'year', 'lag_1', 'lag_7']
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
        imp_df = imp_df.sort_values('Importance', ascending=False)
        st.plotly_chart(px.bar(imp_df.head(10), x='Importance', y='Feature', orientation='h'), use_container_width=True)

    # Inventory
    st.subheader("📦 Inventory Recommendation")
    avg_fc = np.mean(forecast_values)
    safety = 1.65 * np.std(forecast_values) * np.sqrt(7)
    reorder = int(avg_fc * 7 + safety)

    col1, col2, col3 = st.columns(3)
    col1.metric("Recommended Reorder", f"{reorder:,} units")
    col2.metric("Safety Stock", f"{int(safety):,} units")
    col3.metric("Avg Daily Forecast", f"{avg_fc:.1f} units")

    # Download
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecasted_Sales": forecast_values})
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast CSV", csv, f"forecast_store{store}_{family}.csv", "text/csv")

    st.caption(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()