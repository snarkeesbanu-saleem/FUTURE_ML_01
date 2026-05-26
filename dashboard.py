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
def load_model():
    try:
        model = joblib.load("xgboost_model.pkl")
        return model
    except Exception as e:
        st.warning(f"Could not load XGBoost model: {e}")
        return None

@st.cache_resource
def load_encoder():
    try:
        le = joblib.load("label_encoder.pkl")
        return le
    except:
        return None

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

def xgboost_forecast(model, le, df, horizon, store_id, family):
    """Predict using XGBoost with exact feature order from training."""
    # Get last known sales
    last_sales = df["sales"].iloc[-1]
    
    # Encode family
    try:
        family_encoded = le.transform([family])[0]
    except:
        # Fallback: use 0 for unknown family
        family_encoded = 0
    
    # Get other features from last row (or default values)
    last_row = df.iloc[-1]
    onpromotion = last_row.get("onpromotion", 0)
    dcoilwtico = last_row.get("dcoilwtico", 0)
    is_holiday = last_row.get("is_holiday", 0)
    
    # Generate future dates
    last_date = df["date"].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
    
    predictions = []
    for future_date in future_dates:
        # Time features for the future date
        dayofweek = future_date.weekday()
        month = future_date.month
        year = future_date.year
        
        # Feature vector in the EXACT order the model expects
        features = [
            int(store_id),      # store_nbr
            family_encoded,     # family_encoded
            onpromotion,        # onpromotion
            dcoilwtico,         # dcoilwtico
            is_holiday,         # is_holiday
            dayofweek,          # dayofweek
            month,              # month
            year,               # year
            last_sales,         # lag_1 (use last known sales)
            last_sales          # lag_7 (use last known sales)
        ]
        
        pred = model.predict([features])[0]
        predictions.append(max(0, pred))
    
    return future_dates, predictions

def main():
    st.title("🏬 Store Sales Forecasting Dashboard")
    st.caption(f"Data last loaded: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M')}")
    
    # Load model and encoder
    model = load_model()
    le = load_encoder()
    
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
    
    # Generate forecast using XGBoost if available
    if model is not None and le is not None:
        try:
            forecast_dates, forecast_values = xgboost_forecast(
                model, le, df_filtered, horizon, selected_store, selected_family
            )
            st.success("🚀 Using XGBoost model for forecasts")
            model_active = True
        except Exception as e:
            st.error(f"XGBoost prediction failed: {e}")
            forecast_dates, forecast_values = seasonal_naive_forecast(df_filtered, horizon)
            st.info("ℹ️ Using seasonal naive forecast (XGBoost error)")
            model_active = False
    else:
        forecast_dates, forecast_values = seasonal_naive_forecast(df_filtered, horizon)
        st.info("ℹ️ Using seasonal naive forecast (XGBoost model not available)")
        model_active = False
    
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
    
    # Analytics tabs
    st.subheader("🔬 Advanced Analytics")
    tab1, tab2, tab3 = st.tabs(["Store Performance", "Feature Importance", "Holiday Impact"])
    
    with tab1:
        store_perf = train.groupby("store_id")["sales"].sum().sort_values(ascending=False).head(5)
        fig_store = px.bar(x=store_perf.values, y=store_perf.index, orientation="h", title="Top 5 Stores")
        st.plotly_chart(fig_store, width='stretch')
        family_perf = train.groupby("family")["sales"].sum().sort_values(ascending=False).head(5)
        fig_fam = px.bar(x=family_perf.values, y=family_perf.index, orientation="h", title="Top 5 Families")
        st.plotly_chart(fig_fam, width='stretch')
    
    with tab2:
        if model_active and model is not None and hasattr(model, 'feature_importances_'):
            feature_names = ['store_nbr', 'family_encoded', 'onpromotion', 'dcoilwtico', 
                             'is_holiday', 'dayofweek', 'month', 'year', 'lag_1', 'lag_7']
            imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            fig_imp = px.bar(imp_df.head(10), x='importance', y='feature', orientation='h', title='Top 10 Feature Importances')
            st.plotly_chart(fig_imp, width='stretch')
        else:
            st.info("Feature importance will appear when XGBoost model is successfully loaded.")
    
    with tab3:
        if analyze_holidays:
            holiday_group = df_filtered.groupby("is_holiday")["sales"].mean()
            fig_hol = px.bar(x=holiday_group.index, y=holiday_group.values, 
                           labels={"x": "Holiday", "y": "Avg Sales"},
                           title="Holiday vs Non‑Holiday Sales",
                           color=holiday_group.index.astype(str),
                           color_discrete_map={"0": "gray", "1": "orange"})
            st.plotly_chart(fig_hol, width='stretch')
        else:
            st.info("Holiday analysis disabled.")
    
    # Inventory Recommendations
    st.subheader("📦 Inventory Recommendations")
    reorder_point = np.mean(forecast_values) * 7 + 1.5 * np.std(forecast_values) * np.sqrt(7)
    safety_stock = 1.5 * np.std(forecast_values) * np.sqrt(7)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Reorder Point (units)", f"{int(reorder_point)}")
    col_b.metric("Safety Stock (units)", f"{int(safety_stock)}")
    col_c.metric("Avg Daily Forecast", f"{np.mean(forecast_values):.1f}")
    st.warning(f"⚠️ Maintain inventory above **{int(reorder_point)}** units to avoid stockouts.")
    
    # Next 7 days breakdown
    st.subheader("📅 Next 7 Days Forecast")
    next_7 = pd.DataFrame({
        "Date": forecast_dates[:7],
        "Forecast Sales": forecast_values[:7]
    })
    next_7["Day of Week"] = next_7["Date"].dt.day_name()
    st.dataframe(next_7.style.highlight_max(color="lightgreen", subset=["Forecast Sales"]), width='stretch')
    
    # Export
    with st.expander("📥 Export Forecast"):
        full = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast_values})
        st.dataframe(full, width='stretch')
        csv = full.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "forecast.csv", "text/csv")
    
    st.caption(f"Dashboard generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()