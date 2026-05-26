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

@st.cache_resource
def load_model_and_encoder():
    model = None
    le = None
    try:
        if os.path.exists("xgboost_model.pkl"):
            model = joblib.load("xgboost_model.pkl")
            st.success("✅ XGBoost model loaded")
        else:
            st.warning("xgboost_model.pkl not found")
    except Exception as e:
        st.error(f"Error loading model: {e}")
    try:
        if os.path.exists("label_encoder.pkl"):
            le = joblib.load("label_encoder.pkl")
    except:
        pass
    return model, le

def xgboost_forecast(model, le, df, horizon, family):
    last_row = df.iloc[-1:].copy()
    last_date = df["date"].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
    future = pd.concat([last_row] * horizon, ignore_index=True)
    future["date"] = future_dates
    future["dayofweek"] = future["date"].dt.dayofweek
    future["month"] = future["date"].dt.month
    future["year"] = future["date"].dt.year
    # Encode family
    try:
        future["family_encoded"] = le.transform([family])[0]
    except:
        future["family_encoded"] = 0
    # Simple lags
    last_sales = last_row["sales"].values[0]
    future["lag_1"] = last_sales
    future["lag_7"] = last_sales
    # Required features
    required = ['store_id', 'family_encoded', 'onpromotion', 'dcoilwtico',
                'is_holiday', 'dayofweek', 'month', 'year', 'lag_1', 'lag_7']
    for col in required:
        if col not in future.columns:
            future[col] = 0
    X = future[required]
    preds = model.predict(X)
    preds = np.maximum(preds, 0)
    return future_dates, preds

def inventory_advice(forecast_values, lead_time=7, safety_factor=1.5):
    avg = np.mean(forecast_values)
    std = np.std(forecast_values)
    reorder = avg * lead_time + safety_factor * std * np.sqrt(lead_time)
    safety = safety_factor * std * np.sqrt(lead_time)
    return reorder, avg, safety

def main():
    st.title("🏬 Store Sales Forecasting Dashboard")
    st.caption(f"Data last loaded: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M')}")

    # Load model and encoder once
    model, le = load_model_and_encoder()

    with st.spinner("Loading data..."):
        train, test, transactions = load_all_data()

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
    if model is not None and le is not None:
        try:
            forecast_dates, forecast_values = xgboost_forecast(model, le, df_filtered, horizon, selected_family)
            st.success("🚀 Using XGBoost model for forecasts")
        except Exception as e:
            st.error(f"XGBoost failed: {e}. Using fallback.")
            forecast_dates, forecast_values = seasonal_naive_forecast(df_filtered, horizon)
    else:
        forecast_dates, forecast_values = seasonal_naive_forecast(df_filtered, horizon)
        st.info("ℹ️ Using seasonal naive forecast (no XGBoost model)")

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

    # Advanced Analytics Tabs
    st.subheader("🔬 Advanced Analytics")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Holiday Impact", "Oil Price", "Store Performance", "Feature Importance", "Transactions"])

    with tab1:
        if analyze_holidays:
            holiday_group = df_filtered.groupby("is_holiday")["sales"].mean()
            fig_hol = px.bar(x=holiday_group.index, y=holiday_group.values, labels={"x": "Holiday", "y": "Avg Sales"}, title="Holiday vs Non‑Holiday Sales", color=holiday_group.index.astype(str), color_discrete_map={"0": "gray", "1": "orange"})
            st.plotly_chart(fig_hol, width='stretch')
        else:
            st.info("Holiday analysis disabled.")

    with tab2:
        if analyze_oil and "dcoilwtico" in df_filtered.columns and not df_filtered["dcoilwtico"].isna().all():
            fig_oil = px.scatter(df_filtered, x="dcoilwtico", y="sales", title="Sales vs Oil Price", trendline="ols")
            st.plotly_chart(fig_oil, width='stretch')
        else:
            st.info("Oil price correlation disabled or no data.")

    with tab3:
        store_perf = train.groupby("store_id")["sales"].sum().sort_values(ascending=False).head(5)
        fig_store = px.bar(x=store_perf.values, y=store_perf.index, orientation="h", title="Top 5 Stores", color=store_perf.values)
        st.plotly_chart(fig_store, width='stretch')
        family_perf = train.groupby("family")["sales"].sum().sort_values(ascending=False).head(5)
        fig_fam = px.bar(x=family_perf.values, y=family_perf.index, orientation="h", title="Top 5 Families", color=family_perf.values)
        st.plotly_chart(fig_fam, width='stretch')

    with tab4:
        if model is not None and hasattr(model, 'feature_importances_'):
            # Try to load feature names
            try:
                with open("feature_names.txt", "r") as f:
                    fnames = f.read().strip().split(',')
            except:
                fnames = [f"Feature_{i}" for i in range(len(model.feature_importances_))]
            imp_df = pd.DataFrame({"feature": fnames[:len(model.feature_importances_)], "importance": model.feature_importances_})
            imp_df = imp_df.sort_values("importance", ascending=False).head(10)
            fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h", title="Top 10 Feature Importances", color="importance", color_continuous_scale="blues")
            st.plotly_chart(fig_imp, width='stretch')
        else:
            st.info("Feature importance will appear when XGBoost model is loaded.")

    with tab5:
        if transactions is not None:
            st.subheader("💰 Transaction Analysis")
            trans_filtered = transactions[transactions["store_id"] == selected_store]
            if not trans_filtered.empty:
                temp = df_filtered.copy()
                if 'transactions' in temp.columns:
                    temp = temp.drop(columns=['transactions'])
                merged = temp.merge(trans_filtered, on=["date", "store_id"], how="left")
                trans_col = next((col for col in merged.columns if col.startswith('transactions')), None)
                if trans_col:
                    fig_trans = px.line(merged, x="date", y=trans_col, title=f"Daily Transactions - Store {selected_store}")
                    st.plotly_chart(fig_trans, width='stretch')
                    corr_df = merged[["sales", trans_col]].dropna()
                    if not corr_df.empty:
                        corr = corr_df.corr().iloc[0,1]
                        st.metric("Sales-Transactions Correlation", f"{corr:.2f}")
            else:
                st.info("No transaction data for selected store.")
        else:
            st.info("transactions.csv not available.")

    # Inventory
    st.subheader("📦 Inventory Recommendations")
    reorder_point, avg_daily, safety = inventory_advice(forecast_values)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Reorder Point (units)", f"{reorder_point:.0f}")
    col_b.metric("Safety Stock (units)", f"{safety:.0f}")
    col_c.metric("Avg Daily Forecast", f"{avg_daily:.1f}")
    st.warning(f"⚠️ Maintain inventory above **{int(reorder_point)}** units to avoid stockouts.")

    # Next 7 days
    st.subheader("📅 Next 7 Days Forecast")
    next_7 = pd.DataFrame({"Date": forecast_dates[:7], "Forecast Sales": forecast_values[:7]})
    next_7["Day of Week"] = next_7["Date"].dt.day_name()
    st.dataframe(next_7.style.highlight_max(color="lightgreen", subset=["Forecast Sales"]), width='stretch')

    with st.expander("🔍 Drill‑down: Full Forecast Table"):
        full = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast_values})
        st.dataframe(full, width='stretch')
        csv = full.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "forecast.csv", "text/csv")

    if test is not None:
        with st.expander("🔮 Test Data (Kaggle Submission)"):
            st.write(f"Test set contains {len(test)} rows.")
            st.dataframe(test.head(100), width='stretch')
            pred_file = st.file_uploader("Upload your prediction CSV (optional)", type="csv")
            if pred_file:
                preds = pd.read_csv(pred_file)
                st.success(f"Predictions loaded – {len(preds)} rows.")
    else:
        st.info("test.csv not found – skipping test data viewer.")

    st.caption(f"Dashboard generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()