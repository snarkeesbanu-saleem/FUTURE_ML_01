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
REQUIRED_FILES = [f"{DATA_FOLDER}/train.csv", f"{DATA_FOLDER}/stores.csv", 
                  f"{DATA_FOLDER}/oil.csv", f"{DATA_FOLDER}/holidays_events.csv"]

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

@st.cache_data(ttl=3600)
def load_test():
    path = f"{DATA_FOLDER}/test.csv"
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    return None

@st.cache_data(ttl=3600)
def load_transactions():
    path = f"{DATA_FOLDER}/transactions.csv"
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    return None

def load_all_data():
    train = load_train()
    stores = load_stores()
    oil = load_oil()
    holidays = load_holidays()
    test = load_test()
    transactions = load_transactions()

    train = train.merge(stores, on="store_nbr", how="left")
    train = train.merge(oil, on="date", how="left")

    if transactions is not None:
        train.rename(columns={"store_nbr": "store_id"}, inplace=True)
        transactions.rename(columns={"store_nbr": "store_id"}, inplace=True)
        if 'transactions' in train.columns:
            train = train.drop(columns=['transactions'])
        train = train.merge(transactions, on=["date", "store_id"], how="left")
    else:
        train.rename(columns={"store_nbr": "store_id"}, inplace=True)

    train["is_holiday"] = train["date"].isin(holidays["date"]).astype(int)
    return train, test, transactions

# ====================== XGBoost ======================
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
        return seasonal_naive_forecast(df, horizon, store_id, family)
    
    # XGBoost Logic
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
        features = np.array([[
            store_id, family_encoded,
            int(last_row.get('onpromotion', 0)),
            float(last_row.get('dcoilwtico', 0)),
            int(last_row.get('is_holiday', 0)),
            future_date.weekday(), future_date.month, future_date.year,
            last_sales, last_sales
        ]])
        pred = model.predict(features)[0]
        predictions.append(max(0, float(pred)))

    return future_dates, predictions

def seasonal_naive_forecast(df, horizon, store_id, family):
    hist = df.tail(60)["sales"].values
    last_week = hist[-7:] if len(hist) >= 7 else hist
    forecast = [last_week[i % len(last_week)] for i in range(horizon)]
    forecast_dates = [df["date"].max() + timedelta(days=i+1) for i in range(horizon)]
    return forecast_dates, np.maximum(forecast, 0)

def inventory_advice(forecast_values, lead_time=7, safety_factor=1.5):
    avg = np.mean(forecast_values)
    std = np.std(forecast_values)
    reorder = avg * lead_time + safety_factor * std * np.sqrt(lead_time)
    safety = safety_factor * std * np.sqrt(lead_time)
    return reorder, avg, safety

# ====================== MAIN ======================
def main():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        st.error(f"❌ Missing required files: {missing}")
        st.stop()

    st.title("🏬 Store Sales Forecasting Dashboard")
    st.caption(f"Data last loaded: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M')}")

    with st.spinner("Loading large dataset..."):
        train, test, transactions = load_all_data()

    model = load_model()
    le = load_label_encoder()

    # ... [Rest of your original main() function remains the same] ...
    # (Sidebar, filters, charts, tabs, inventory, etc.)

    st.sidebar.header("🔍 Filters")
    with st.sidebar.form(key="filters_form"):
        stores_list = sorted(train["store_id"].unique())
        families = sorted(train["family"].unique())
        selected_store = st.selectbox("Store", stores_list)
        selected_family = st.selectbox("Product Family", families)
        horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)

        date_min = train["date"].min().date()
        date_max = train["date"].max().date()
        date_range = st.date_input("Date range", [date_min, date_max])

        with st.expander("⚙️ Advanced"):
            show_ci = st.checkbox("Show confidence intervals", True)
            analyze_holidays = st.checkbox("Holiday impact", True)
            analyze_oil = st.checkbox("Oil price correlation", True)

        submitted = st.form_submit_button("Apply Filters")

    if not submitted:
        st.stop()

    # Filter data
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1])
    df_filtered = train[
        (train["store_id"] == selected_store) &
        (train["family"] == selected_family) &
        (train["date"] >= start) & (train["date"] <= end)
    ].copy()

    if df_filtered.empty:
        st.error("No data for these filters.")
        st.stop()

    forecast_dates, forecast_values = get_forecast(model, le, df_filtered, horizon, selected_store, selected_family)

    # KPIs, Chart, Tabs, etc. (your original code continues here)
    # ... Paste the rest of your original dashboard code from KPIs down ...

    # (For brevity, I can send the full file if needed)

if __name__ == "__main__":
    main()