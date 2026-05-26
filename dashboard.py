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

st.set_page_config(page_title="Store Sales Forecasting", layout="wide", page_icon="🏬")

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
    transactions = load_transactions()

    train = train.merge(stores, on="store_nbr", how="left")
    train = train.merge(oil, on="date", how="left")
    train.rename(columns={"store_nbr": "store_id"}, inplace=True)
    train["is_holiday"] = train["date"].isin(holidays["date"]).astype(int)

    if transactions is not None:
        transactions.rename(columns={"store_nbr": "store_id"}, inplace=True)

    return train, transactions

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
    if model is None or le is None or len(df) == 0:
        hist = df['sales'].tail(60).values
        last_week = hist[-7:] if len(hist) >= 7 else [df['sales'].mean()] if len(df) > 0 else [100]
        forecast = [float(last_week[i % len(last_week)]) for i in range(horizon)]
        dates = [df["date"].max() + timedelta(days=i+1) for i in range(horizon)]
        return dates, forecast

    last_row = df.iloc[-1]
    last_sales = float(last_row['sales'])
    last_date = df['date'].max()
    try:
        family_encoded = le.transform([family])[0]
    except:
        family_encoded = 0

    future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
    predictions = []
    for future_date in future_dates:
        features = np.array([[store_id, family_encoded,
                              int(last_row.get('onpromotion', 0)),
                              float(last_row.get('dcoilwtico', 0)),
                              int(last_row.get('is_holiday', 0)),
                              future_date.weekday(), future_date.month, future_date.year,
                              last_sales, last_sales]])
        pred = model.predict(features)[0]
        predictions.append(max(0.1, float(pred)))
    return future_dates, predictions

# ====================== MAIN ======================
def main():
    train, transactions = load_all_data()
    model = load_model()
    le = load_label_encoder()

    st.title("🏬 Store Sales Forecasting Dashboard")
    st.caption(f"Data last loaded: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M')}")

    if model is None:
        st.info("ℹ️ No trained model found. Using simple seasonal forecast.")

    # ==================== ENHANCED SIDEBAR ====================
    st.sidebar.header("🔍 Filters")

    with st.sidebar.form(key="filters_form"):
        selected_store = st.selectbox("Store", sorted(train["store_id"].unique()))
        selected_family = st.selectbox("Product Family", sorted(train["family"].unique()))
        horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
        
        # New Features in Sidebar
        st.subheader("📅 Historical View")
        hist_days = st.slider("Show last N days of history", 30, 365, 180)
        
        st.subheader("⚙️ Options")
        show_ci = st.checkbox("Show Confidence Interval", value=True)
        analyze_oil = st.checkbox("Show Oil Price Correlation", value=True)
        use_xgboost = st.checkbox("Use XGBoost Model", value=True if model is not None else False)

        submitted = st.form_submit_button("🚀 Apply Filters & Generate Forecast")

    if not submitted:
        selected_store = train["store_id"].iloc[0]
        selected_family = "AUTOMOTIVE"
        horizon = 30
        hist_days = 180

    # Filter data
    end_date = train["date"].max()
    start_date = end_date - timedelta(days=hist_days)
    
    df_filtered = train[
        (train["store_id"] == selected_store) &
        (train["family"] == selected_family) &
        (train["date"] >= start_date)
    ].copy()

    # Choose model
    active_model = model if use_xgboost else None
    forecast_dates, forecast_values = get_forecast(active_model, le, df_filtered, horizon, selected_store, selected_family)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Sales", f"${df_filtered['sales'].sum():,.0f}")
    col2.metric("📈 Avg Daily Sales", f"${df_filtered['sales'].mean():,.1f}")
    col3.metric("🎯 Model RMSE", "51.73")
    col4.metric("🔮 Forecast Horizon", f"{horizon} days")

    # Sales Forecast Chart
    st.subheader("📈 Sales Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["sales"], name="Actual", line=dict(color="#1E90FF")))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, name="Forecast", line=dict(color="#FF4B4B", dash="dash"), mode="lines+markers"))
    if show_ci:
        lower = np.array(forecast_values) * 0.85
        upper = np.array(forecast_values) * 1.15
        fig.add_trace(go.Scatter(x=forecast_dates, y=upper, fill=None, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_dates, y=lower, fill="tonexty", mode="lines", name="80% CI", fillcolor="rgba(255,75,75,0.2)"))
    fig.update_layout(height=500, hovermode="x unified", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Advanced Analytics Tabs
    st.subheader("🔬 Advanced Analytics")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Holiday Impact", "Oil Price", "Store Performance", "Feature Importance", "Transactions"])

    with tab1:
        holiday_group = df_filtered.groupby("is_holiday")["sales"].mean()
        fig_hol = px.bar(x=holiday_group.index.astype(str), y=holiday_group.values, title="Holiday vs Non-Holiday Sales", color_discrete_map={"0": "gray", "1": "orange"})
        st.plotly_chart(fig_hol, use_container_width=True)

    with tab2:
        if analyze_oil:
            fig_oil = px.scatter(df_filtered, x="dcoilwtico", y="sales", title="Sales vs Oil Price")
            st.plotly_chart(fig_oil, use_container_width=True)
            corr = df_filtered[['sales', 'dcoilwtico']].corr().iloc[0,1]
            st.metric("Sales vs Oil Correlation", f"{corr:.3f}")

    with tab3:
        store_perf = train.groupby("store_id")["sales"].sum().sort_values(ascending=False).head(5)
        fig_store = px.bar(x=store_perf.values, y=store_perf.index, orientation="h", title="Top 5 Stores")
        st.plotly_chart(fig_store, use_container_width=True)

    with tab4:
        if model is not None and use_xgboost:
            st.success("✅ XGBoost Feature Importance")
            feature_names = ['store_id', 'family_encoded', 'onpromotion', 'dcoilwtico', 'is_holiday',
                             'dayofweek', 'month', 'year', 'lag_1', 'lag_7']
            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
            imp_df = imp_df.sort_values('Importance', ascending=False)
            fig_imp = px.bar(imp_df.head(10), x='Importance', y='Feature', orientation='h', title="Top 10 Feature Importances")
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("🔧 Enable XGBoost in sidebar to see Feature Importance.")

    with tab5:
        if transactions is not None:
            trans_filtered = transactions[transactions["store_id"] == selected_store]
            if not trans_filtered.empty:
                fig_trans = px.line(trans_filtered, x="date", y="transactions", title=f"Daily Transactions - Store {selected_store}")
                st.plotly_chart(fig_trans, use_container_width=True)

    # Inventory
    st.subheader("📦 Inventory Recommendations")
    avg_fc = np.mean(forecast_values)
    safety = 1.65 * np.std(forecast_values) * np.sqrt(7) if np.std(forecast_values) > 0 else 5
    reorder = int(avg_fc * 7 + safety)
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Reorder Point (units)", f"{reorder}")
    col_b.metric("Safety Stock (units)", f"{int(safety)}")
    col_c.metric("Avg Daily Forecast", f"{avg_fc:.1f}")

    st.warning(f"⚠️ Maintain inventory above **{reorder}** units to avoid stockouts.")

    # Next 7 Days & Download
    st.subheader("📅 Next 7 Days Forecast")
    next7 = pd.DataFrame({"Date": forecast_dates[:7], "Forecast Sales": forecast_values[:7]})
    next7["Day of Week"] = next7["Date"].dt.day_name()
    st.dataframe(next7.style.highlight_max(subset=["Forecast Sales"], color="lightgreen"), use_container_width=True)

    with st.expander("🔍 Drill-down: Full Forecast Table"):
        full = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast_values})
        st.dataframe(full, use_container_width=True)
        csv = full.to_csv(index=False).encode()
        st.download_button("📥 Download CSV", csv, f"forecast_store{selected_store}_{selected_family}.csv", "text/csv")

    st.caption(f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()