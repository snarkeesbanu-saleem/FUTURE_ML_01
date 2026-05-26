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

# ------------------------------------------------------------------
# Helper to find files (root or subfolder)
# ------------------------------------------------------------------
def find_file(filename):
    if os.path.exists(filename):
        return filename
    subfolder = f"store-sales-time-series-forecasting (2)/{filename}"
    if os.path.exists(subfolder):
        return subfolder
    return None

# ------------------------------------------------------------------
# Load data with caching
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# XGBoost model loading (with caching)
# ------------------------------------------------------------------
@st.cache_resource
def load_xgboost_model():
    try:
        if os.path.exists("xgboost_model.pkl"):
            model = joblib.load("xgboost_model.pkl")
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}")
        return None

@st.cache_resource
def load_label_encoder():
    try:
        if os.path.exists("label_encoder.pkl"):
            le = joblib.load("label_encoder.pkl")
            return le
        else:
            return None
    except Exception:
        return None

# ------------------------------------------------------------------
# Forecast functions
# ------------------------------------------------------------------
def seasonal_forecast(df, horizon):
    hist = df.tail(60)["sales"].values
    if len(hist) == 0:
        forecast = [0] * horizon
    else:
        last_week = hist[-7:] if len(hist) >= 7 else hist
        forecast = [last_week[i % len(last_week)] for i in range(horizon)]
    forecast = np.maximum(forecast, 0)
    forecast_dates = [df["date"].max() + timedelta(days=i+1) for i in range(horizon)]
    return forecast_dates, forecast

def xgboost_forecast(model, le, df, horizon, store_id, family):
    """
    Generate forecast using trained XGBoost model.
    Features must match training exactly.
    """
    # Last known sales
    last_sales = df["sales"].iloc[-1]
    
    # Encode family
    try:
        family_encoded = le.transform([family])[0]
    except:
        # Fallback if family not seen during training
        family_encoded = 0
    
    # Get last row features
    last_row = df.iloc[-1]
    onpromotion = last_row.get("onpromotion", 0)
    dcoilwtico = last_row.get("dcoilwtico", 0) or 0
    is_holiday = last_row.get("is_holiday", 0)
    
    last_date = df["date"].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
    
    predictions = []
    for future_date in future_dates:
        dayofweek = future_date.weekday()
        month = future_date.month
        year = future_date.year
        
        # Feature vector in the EXACT order as training
        features = [
            int(store_id),      # store_nbr
            family_encoded,     # family_encoded
            onpromotion,        # onpromotion
            dcoilwtico,         # dcoilwtico
            is_holiday,         # is_holiday
            dayofweek,          # dayofweek
            month,              # month
            year,               # year
            last_sales,         # lag_1
            last_sales          # lag_7
        ]
        
        pred = model.predict([features])[0]
        predictions.append(max(0, pred))
    
    return future_dates, predictions

# ------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------
def main():
    st.title("🏬 Store Sales Forecasting Dashboard")
    st.caption(f"Data last loaded: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M')}")
    
    # Load data
    with st.spinner("Loading data..."):
        train, test, transactions = load_all_data()
    
    # Load model and encoder
    xgb_model = load_xgboost_model()
    label_encoder = load_label_encoder()
    
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
        
        show_ci = st.checkbox("Show confidence intervals", True)
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
    
    # Try XGBoost forecast
    use_xgboost = False
    try:
        if xgb_model is not None and label_encoder is not None:
            forecast_dates, forecast_values = xgboost_forecast(
                xgb_model, label_encoder, df_filtered, horizon, selected_store, selected_family
            )
            use_xgboost = True
            st.success("🚀 Using XGBoost model for forecasts")
        else:
            raise ValueError("Model or encoder missing")
    except Exception as e:
        st.warning(f"XGBoost forecast failed: {e}. Using seasonal fallback.")
        forecast_dates, forecast_values = seasonal_forecast(df_filtered, horizon)
        use_xgboost = False
    
    if not use_xgboost:
        st.info("📈 Using seasonal naive forecast (XGBoost not available)")
    
    # KPIs
    total_sales = df_filtered["sales"].sum()
    avg_sales = df_filtered["sales"].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
    col2.metric("📈 Avg Daily Sales", f"${avg_sales:,.0f}")
    col3.metric("🔮 Forecast Horizon", f"{horizon} days")
    
    # Forecast chart
    st.subheader("📈 Sales Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["sales"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, mode="lines+markers", name="Forecast", line=dict(dash="dash")))
    if show_ci:
        lower = np.array(forecast_values) * 0.85
        upper = np.array(forecast_values) * 1.15
        fig.add_trace(go.Scatter(x=forecast_dates, y=upper, fill=None, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_dates, y=lower, fill="tonexty", mode="lines", line=dict(width=0), name="80% CI", fillcolor="rgba(255,0,0,0.2)"))
    fig.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(fig, width='stretch')
    
    # Feature importance (if XGBoost)
    if use_xgboost and xgb_model is not None and hasattr(xgb_model, 'feature_importances_'):
        st.subheader("📊 Feature Importance")
        feature_names = ['store_nbr', 'family_encoded', 'onpromotion', 'dcoilwtico', 
                         'is_holiday', 'dayofweek', 'month', 'year', 'lag_1', 'lag_7']
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        fig_imp = px.bar(imp_df, x='importance', y='feature', orientation='h', title='Top Features', color='importance', color_continuous_scale='blues')
        st.plotly_chart(fig_imp, width='stretch')
    
    # Inventory recommendation
    st.subheader("📦 Inventory Recommendations")
    reorder = np.mean(forecast_values) * 7 + 1.5 * np.std(forecast_values) * np.sqrt(7)
    safety = 1.5 * np.std(forecast_values) * np.sqrt(7)
    col_a, col_b = st.columns(2)
    col_a.metric("Reorder Point (units)", f"{int(reorder)}")
    col_b.metric("Safety Stock (units)", f"{int(safety)}")
    st.warning(f"⚠️ Maintain inventory above **{int(reorder)}** units to avoid stockouts.")
    
    # Export forecast
    with st.expander("📥 Export Forecast"):
        export_df = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast_values})
        st.dataframe(export_df)
        csv = export_df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "forecast.csv", "text/csv")
    
    st.caption(f"Dashboard generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()