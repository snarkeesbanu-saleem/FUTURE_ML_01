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

def seasonal_naive_forecast(df, horizon):
    hist = df.tail(60)["sales"].values
    if len(hist) == 0:
        forecast = np.zeros(horizon)
    else:
        last_week = hist[-7:] if len(hist) >= 7 else hist
        forecast = [last_week[i % len(last_week)] for i in range(horizon)]
    forecast = np.maximum(forecast, 0)
    forecast_dates = [df["date"].max() + timedelta(days=i+1) for i in range(horizon)]
    return forecast_dates, forecast

def main():
    st.title("🏬 Store Sales Forecasting Dashboard")
    st.caption(f"Data last loaded: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M')}")
    
    # ---- DEBUG INFO ----
    with st.expander("🔧 Debug Info (click to expand)"):
        st.write("Current directory:", os.getcwd())
        files = os.listdir(".")
        st.write("Files in directory:", files)
        if "xgboost_model.pkl" in files:
            st.success("✅ xgboost_model.pkl found")
            size = os.path.getsize("xgboost_model.pkl") / 1024
            st.write(f"   Size: {size:.1f} KB")
        else:
            st.error("❌ xgboost_model.pkl NOT found")
        if "label_encoder.pkl" in files:
            st.success("✅ label_encoder.pkl found")
        else:
            st.error("❌ label_encoder.pkl NOT found")
    
    # ---- LOAD DATA ----
    with st.spinner("Loading data..."):
        try:
            train, test, transactions = load_all_data()
            st.success("Data loaded successfully")
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()
    
    # ---- LOAD MODEL ----
    model = None
    le = None
    model_error = None
    try:
        if os.path.exists("xgboost_model.pkl"):
            model = joblib.load("xgboost_model.pkl")
            st.success("✅ XGBoost model loaded")
        else:
            st.warning("xgboost_model.pkl not found")
    except Exception as e:
        model_error = f"Model load error: {e}"
        st.error(model_error)
    
    try:
        if os.path.exists("label_encoder.pkl"):
            le = joblib.load("label_encoder.pkl")
            st.success("✅ Label encoder loaded")
        else:
            st.warning("label_encoder.pkl not found")
    except Exception as e:
        st.error(f"Encoder load error: {e}")
    
    # ---- SIDEBAR FILTERS ----
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
        
        submitted = st.form_submit_button("Apply Filters")
    
    if not submitted:
        st.stop()
    
    # ---- FILTER DATA ----
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
    
    # ---- FORECAST ----
    forecast_dates = None
    forecast_values = None
    using_xgboost = False
    
    if model is not None and le is not None and model_error is None:
        try:
            # Get last sales
            last_sales = df_filtered["sales"].iloc[-1]
            
            # Encode family
            try:
                family_encoded = le.transform([selected_family])[0]
            except:
                family_encoded = 0
            
            # Get last row features
            last_row = df_filtered.iloc[-1]
            onpromotion = last_row.get("onpromotion", 0)
            dcoilwtico = last_row.get("dcoilwtico", 0)
            is_holiday = last_row.get("is_holiday", 0)
            
            # Generate future dates
            last_date = df_filtered["date"].max()
            future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
            
            predictions = []
            for future_date in future_dates:
                dayofweek = future_date.weekday()
                month = future_date.month
                year = future_date.year
                
                # Feature array
                features = [
                    int(selected_store),  # store_nbr
                    family_encoded,       # family_encoded
                    onpromotion,          # onpromotion
                    dcoilwtico,           # dcoilwtico
                    is_holiday,           # is_holiday
                    dayofweek,            # dayofweek
                    month,                # month
                    year,                 # year
                    last_sales,           # lag_1
                    last_sales            # lag_7
                ]
                
                pred = model.predict([features])[0]
                predictions.append(max(0, pred))
            
            forecast_dates = future_dates
            forecast_values = predictions
            using_xgboost = True
            st.success("🚀 Using XGBoost model for forecasts")
        except Exception as e:
            st.error(f"XGBoost prediction failed: {e}")
            # Fallback
            forecast_dates, forecast_values = seasonal_naive_forecast(df_filtered, horizon)
            st.info("ℹ️ Using seasonal naive forecast (fallback)")
    else:
        forecast_dates, forecast_values = seasonal_naive_forecast(df_filtered, horizon)
        st.info("ℹ️ Using seasonal naive forecast (XGBoost model not available)")
    
    # ---- KPIS ----
    total_sales = df_filtered["sales"].sum()
    avg_sales = df_filtered["sales"].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
    col2.metric("📈 Avg Daily Sales", f"${avg_sales:,.0f}")
    col3.metric("🎯 Model", "XGBoost" if using_xgboost else "Seasonal")
    col4.metric("🔮 Horizon", f"{horizon} days")
    
    # ---- CHART ----
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
    
    # ---- FEATURE IMPORTANCE (if XGBoost) ----
    if using_xgboost and model is not None and hasattr(model, 'feature_importances_'):
        st.subheader("📊 Feature Importance")
        feature_names = ['store_nbr', 'family_encoded', 'onpromotion', 'dcoilwtico', 
                         'is_holiday', 'dayofweek', 'month', 'year', 'lag_1', 'lag_7']
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        fig_imp = px.bar(imp_df.head(10), x='importance', y='feature', orientation='h', title='Top Features')
        st.plotly_chart(fig_imp, width='stretch')
    
    # ---- INVENTORY ----
    st.subheader("📦 Inventory Recommendations")
    reorder = np.mean(forecast_values) * 7 + 1.5 * np.std(forecast_values) * np.sqrt(7)
    st.metric("Reorder Point (units)", f"{int(reorder)}")
    st.warning(f"⚠️ Maintain inventory above **{int(reorder)}** units.")
    
    # ---- NEXT 7 DAYS ----
    st.subheader("📅 Next 7 Days Forecast")
    next_7 = pd.DataFrame({
        "Date": forecast_dates[:7],
        "Forecast Sales": forecast_values[:7]
    })
    next_7["Day of Week"] = next_7["Date"].dt.day_name()
    st.dataframe(next_7, width='stretch')
    
    # ---- EXPORT ----
    with st.expander("📥 Export Forecast"):
        export_df = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast_values})
        st.dataframe(export_df)
        csv = export_df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "forecast.csv", "text/csv")

if __name__ == "__main__":
    main()