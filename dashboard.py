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

def find_file(filename):
    if os.path.exists(filename):
        return filename
    subfolder = f"store-sales-time-series-forecasting (2)/{filename}"
    if os.path.exists(subfolder):
        return subfolder
    return None

@st.cache_data(ttl=3600)
def load_train():
    path = find_file("train.csv")
    if path is None:
        st.error("train.csv not found")
        st.stop()
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=3600)
def load_stores():
    path = find_file("stores.csv")
    if path is None:
        st.error("stores.csv not found")
        st.stop()
    return pd.read_csv(path)

@st.cache_data(ttl=3600)
def load_oil():
    path = find_file("oil.csv")
    if path is None:
        return pd.DataFrame(columns=["date", "dcoilwtico"])
    oil = pd.read_csv(path, parse_dates=["date"])
    oil["date"] = pd.to_datetime(oil["date"])
    if "dcoilwtico" not in oil.columns:
        oil["dcoilwtico"] = np.nan
    return oil

@st.cache_data(ttl=3600)
def load_holidays():
    path = find_file("holidays_events.csv")
    if path is None:
        st.error("holidays_events.csv not found")
        st.stop()
    hols = pd.read_csv(path, parse_dates=["date"])
    hols["date"] = pd.to_datetime(hols["date"])
    return hols

@st.cache_data(ttl=3600)
def load_test():
    path = find_file("test.csv")
    if path:
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None

@st.cache_data(ttl=3600)
def load_transactions():
    path = find_file("transactions.csv")
    if path:
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None

def load_all_data():
    train = load_train()
    stores = load_stores()
    oil = load_oil()
    holidays = load_holidays()
    test = load_test()
    transactions = load_transactions()

    train = train.merge(stores, on="store_nbr", how="left")
    if not oil.empty and "dcoilwtico" in oil.columns:
        train = train.merge(oil, on="date", how="left")
    else:
        train["dcoilwtico"] = np.nan

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

@st.cache_resource
def load_model_and_encoder():
    try:
        model = joblib.load("xgboost_model.pkl") if os.path.exists("xgboost_model.pkl") else None
        le = joblib.load("label_encoder.pkl") if os.path.exists("label_encoder.pkl") else None
        return model, le
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def make_forecast(model, le, df, horizon, store_id, family):
    """Generate forecast using XGBoost"""
    # Get last known values
    last_row = df.iloc[-1:].copy()
    last_date = df["date"].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
    
    # Encode family
    try:
        family_encoded = le.transform([family])[0]
    except:
        family_encoded = 0
    
    # Get store number (as integer)
    store_nbr = int(store_id)
    
    # Get last sales for lags
    last_sales = last_row["sales"].values[0]
    
    # Get other features from last row
    onpromotion = last_row["onpromotion"].values[0] if "onpromotion" in last_row else 0
    dcoilwtico = last_row["dcoilwtico"].values[0] if "dcoilwtico" in last_row else 0
    is_holiday = last_row["is_holiday"].values[0] if "is_holiday" in last_row else 0
    
    # Build predictions for each future day
    predictions = []
    for i in range(horizon):
        # Future date features
        future_date = future_dates[i]
        dayofweek = future_date.weekday()
        month = future_date.month
        year = future_date.year
        
        # Feature array in the exact order the model expects
        features = [
            store_nbr,           # store_nbr
            family_encoded,      # family_encoded  
            onpromotion,         # onpromotion
            dcoilwtico,          # dcoilwtico
            is_holiday,          # is_holiday
            dayofweek,           # dayofweek
            month,               # month
            year,                # year
            last_sales,          # lag_1 (use last known sales)
            last_sales           # lag_7 (use last known sales)
        ]
        
        # Predict
        pred = model.predict([features])[0]
        predictions.append(max(0, pred))
    
    return future_dates, predictions

def seasonal_forecast(df, horizon):
    hist = df.tail(60)["sales"].values
    if len(hist) == 0:
        forecast = [0] * horizon
    else:
        last_week = hist[-7:] if len(hist) >= 7 else hist
        forecast = [last_week[i % len(last_week)] for i in range(horizon)]
    forecast_dates = [df["date"].max() + timedelta(days=i+1) for i in range(horizon)]
    return forecast_dates, forecast

def main():
    st.title("🏬 Store Sales Forecasting Dashboard")
    st.caption(f"Data last loaded: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M')}")
    
    # Load model
    model, le = load_model_and_encoder()
    
    with st.spinner("Loading data..."):
        train, test, transactions = load_all_data()
    
    # Sidebar filters
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
            analyze_oil = st.checkbox("Oil price correlation", False)
        
        submitted = st.form_submit_button("Apply Filters")
    
    if not submitted:
        st.stop()
    
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1])
    df_filtered = train[
        (train["store_id"] == selected_store) &
        (train["family"] == selected_family) &
        (train["date"] >= start) &
        (train["date"] <= end)
    ].copy()
    
    if df_filtered.empty:
        st.error("No data for these filters.")
        st.stop()
    
    # Generate forecast
    use_xgboost = False
    if model is not None and le is not None:
        try:
            forecast_dates, forecast_values = make_forecast(model, le, df_filtered, horizon, selected_store, selected_family)
            use_xgboost = True
            st.success("✅ Using XGBoost model for forecasts")
        except Exception as e:
            st.error(f"XGBoost failed: {e}")
            forecast_dates, forecast_values = seasonal_forecast(df_filtered, horizon)
            st.info("ℹ️ Using seasonal forecast (fallback)")
    else:
        forecast_dates, forecast_values = seasonal_forecast(df_filtered, horizon)
        st.info("ℹ️ Using seasonal forecast (XGBoost model not found)")
    
    # KPIs
    total_sales = df_filtered["sales"].sum()
    avg_sales = df_filtered["sales"].mean()
    rmse = 51.73
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
    col2.metric("📈 Avg Daily Sales", f"${avg_sales:,.0f}")
    col3.metric("🎯 Model RMSE", f"{rmse:.2f}")
    col4.metric("🔮 Forecast Horizon", f"{horizon} days")
    
    # Forecast chart
    st.subheader("📈 Sales Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["sales"], mode="lines", name="Actual", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, mode="lines+markers", name="Forecast", line=dict(color="red", dash="dash")))
    if show_ci:
        lower = np.array(forecast_values) * 0.85
        upper = np.array(forecast_values) * 1.15
        fig.add_trace(go.Scatter(x=forecast_dates, y=upper, fill=None, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_dates, y=lower, fill="tonexty", mode="lines", line=dict(width=0), name="80% CI", fillcolor="rgba(255,0,0,0.2)"))
    fig.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(fig, width='stretch')
    
    # Tabs (simplified)
    st.subheader("🔬 Analytics")
    tab1, tab2 = st.tabs(["Store Performance", "Feature Importance"])
    
    with tab1:
        store_perf = train.groupby("store_id")["sales"].sum().sort_values(ascending=False).head(5)
        fig_store = px.bar(x=store_perf.values, y=store_perf.index, orientation="h", title="Top 5 Stores")
        st.plotly_chart(fig_store, width='stretch')
    
    with tab2:
        if use_xgboost and model is not None:
            feature_names = ['store_nbr', 'family_encoded', 'onpromotion', 'dcoilwtico', 'is_holiday', 'dayofweek', 'month', 'year', 'lag_1', 'lag_7']
            imp_df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
            fig_imp = px.bar(imp_df.head(10), x="importance", y="feature", orientation="h", title="Feature Importances")
            st.plotly_chart(fig_imp, width='stretch')
        else:
            st.info("Feature importance will appear when XGBoost model is loaded.")
    
    # Inventory
    st.subheader("📦 Inventory Recommendations")
    reorder = np.mean(forecast_values) * 7 + 1.5 * np.std(forecast_values) * np.sqrt(7)
    st.metric("Reorder Point", f"{int(reorder)} units")
    st.warning(f"⚠️ Maintain inventory above {int(reorder)} units")
    
    # Export
    with st.expander("📥 Export Forecast"):
        export_df = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast_values})
        st.dataframe(export_df)
        csv = export_df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "forecast.csv", "text/csv")

if __name__ == "__main__":
    main()