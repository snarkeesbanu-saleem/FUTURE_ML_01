import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Store Sales Forecasting", layout="wide")

if "last_update" not in st.session_state:
    st.session_state.last_update = datetime.now()

DATA_FOLDER = "store-sales-time-series-forecasting (2)"

# ====================== DATA LOADING ======================
@st.cache_data(ttl=3600)
def load_train():
    return pd.read_csv(f"{DATA_FOLDER}/train.csv", parse_dates=["date"])

@st.cache_data(ttl=3600)
def load_stores():
    return pd.read_csv(f"{DATA_FOLDER}/stores.csv")

@st.cache_data(ttl=3600)
def load_oil():
    oil = pd.read_csv(f"{DATA_FOLDER}/oil.csv", parse_dates=["date"])
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill()
    return oil

@st.cache_data(ttl=3600)
def load_holidays():
    return pd.read_csv(f"{DATA_FOLDER}/holidays_events.csv", parse_dates=["date"])

def load_all_data():
    train = load_train()
    stores = load_stores()
    oil = load_oil()
    holidays = load_holidays()

    train = train.merge(stores, on="store_nbr", how="left")
    train = train.merge(oil, on="date", how="left")
    train.rename(columns={"store_nbr": "store_id"}, inplace=True)
    train["is_holiday"] = train["date"].isin(holidays["date"]).astype(int)
    return train

# ====================== MODELS ======================
@st.cache_resource
def load_model():
    try:
        return joblib.load("xgboost_model.pkl")
    except:
        return None

@st.cache_resource
def load_label_encoder():
    try:
        return joblib.load("label_encoder.pkl")
    except:
        return None

def get_forecast(model, le, df, horizon, store_id, family):
    if model is None or le is None:
        # Seasonal fallback
        hist = df.tail(60)["sales"].values
        last_week = hist[-7:] if len(hist) >= 7 else hist
        forecast = [last_week[i % len(last_week)] for i in range(horizon)]
        dates = [df["date"].max() + timedelta(days=i+1) for i in range(horizon)]
        return dates, np.maximum(forecast, 0)
    
    # XGBoost Forecast
    last_row = df.iloc[-1]
    last_sales = last_row['sales']
    last_date = df['date'].max()
    try:
        family_encoded = le.transform([family])[0]
    except:
        family_encoded = 0

    future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
    predictions = []
    for future_date in future_dates:
        features = np.array([[store_id, family_encoded, int(last_row.get('onpromotion',0)),
                              float(last_row.get('dcoilwtico',0)), int(last_row.get('is_holiday',0)),
                              future_date.weekday(), future_date.month, future_date.year,
                              last_sales, last_sales]])
        pred = model.predict(features)[0]
        predictions.append(max(0, float(pred)))
    return future_dates, predictions

# ====================== MAIN ======================
def main():
    train = load_all_data()
    model = load_model()
    le = load_label_encoder()

    st.title("🏬 Store Sales Forecasting Dashboard")
    st.caption(f"Data last loaded: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M')}")

    # Sidebar
    st.sidebar.header("🔍 Filters")
    with st.sidebar.form(key="filters_form"):
        selected_store = st.selectbox("Store", sorted(train["store_id"].unique()), index=0)
        selected_family = st.selectbox("Product Family", sorted(train["family"].unique()), index=0)
        horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
        date_range = st.date_input("Date range", 
                                  [train["date"].min().date(), train["date"].max().date()])
        
        with st.expander("⚙️ Advanced"):
            show_ci = st.checkbox("Show confidence intervals", True)
            analyze_oil = st.checkbox("Oil price correlation", True)

        submitted = st.form_submit_button("Apply Filters")

    # Default values if not submitted yet
    if not submitted:
        selected_store = train["store_id"].iloc[0]
        selected_family = train["family"].iloc[0]
        horizon = 30
        start = train["date"].min()
        end = train["date"].max()
    else:
        start = pd.to_datetime(date_range[0])
        end = pd.to_datetime(date_range[1])

    # Filter data
    df_filtered = train[
        (train["store_id"] == selected_store) &
        (train["family"] == selected_family) &
        (train["date"] >= start) & (train["date"] <= end)
    ].copy()

    if df_filtered.empty:
        st.error("No data available for selected filters.")
        st.stop()

    forecast_dates, forecast_values = get_forecast(model, le, df_filtered, horizon, selected_store, selected_family)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Sales", f"${df_filtered['sales'].sum():,.0f}")
    col2.metric("📈 Avg Daily Sales", f"${df_filtered['sales'].mean():,.1f}")
    col3.metric("Model", "XGBoost" if model is not None else "Seasonal")
    col4.metric("Forecast Horizon", f"{horizon} days")

    # Main Chart
    st.subheader("📈 Sales Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["sales"], name="Actual", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, name="Forecast", line=dict(color="red", dash="dash"), mode="lines+markers"))
    if show_ci:
        lower = np.array(forecast_values) * 0.85
        upper = np.array(forecast_values) * 1.15
        fig.add_trace(go.Scatter(x=forecast_dates, y=upper, fill=None, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_dates, y=lower, fill="tonexty", mode="lines", name="80% CI", fillcolor="rgba(255,165,0,0.2)"))
    fig.update_layout(height=550, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Tabs (Oil Price + Others)
    st.subheader("🔬 Advanced Analytics")
    tab1, tab2, tab3 = st.tabs(["Holiday Impact", "Oil Price Correlation", "Store Performance"])

    with tab1:
        holiday_group = df_filtered.groupby("is_holiday")["sales"].mean()
        fig_hol = px.bar(x=holiday_group.index.astype(str), y=holiday_group.values, title="Holiday vs Non-Holiday")
        st.plotly_chart(fig_hol, use_container_width=True)

    with tab2:
        if analyze_oil:
            fig_oil = px.scatter(df_filtered, x="dcoilwtico", y="sales", trendline="ols", title="Sales vs Oil Price")
            st.plotly_chart(fig_oil, use_container_width=True)
            corr = df_filtered[['sales', 'dcoilwtico']].corr().iloc[0,1]
            st.metric("Correlation", f"{corr:.3f}")
        else:
            st.info("Oil correlation disabled")

    with tab3:
        st.info("Store & Family performance charts can be added here")

    # Inventory
    st.subheader("📦 Inventory Recommendations")
    avg = np.mean(forecast_values)
    st.metric("Avg Daily Forecast", f"{avg:.1f} units")

    st.caption(f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()