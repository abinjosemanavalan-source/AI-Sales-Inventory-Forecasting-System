
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "data" / "processed"
MODEL_PATH = PROJECT_FOLDER / "models" / "sales_forecast_model_final.pkl"

st.set_page_config(
    page_title="AI Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_FOLDER / "forecasting_dataset.csv",
        parse_dates=["Date"],
        dtype={"StockCode": str}
    )

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
saved = load_model()
model = saved["model"]
features = saved["features"]

st.title("AI Sales & Inventory Forecasting")
st.caption("Historical sales analysis and experimental demand forecasts")

products = sorted(df["StockCode"].unique().tolist())

selected_product = st.selectbox(
    "Select product",
    products,
    index=products.index("85099B") if "85099B" in products else 0
)

product_df = (
    df[df["StockCode"] == selected_product]
    .sort_values("Date")
    .copy()
)

# Dashboard summary
col1, col2, col3 = st.columns(3)

col1.metric("Total units sold", f"{product_df['UnitsSold'].sum():,}")
col2.metric("Average daily units", f"{product_df['UnitsSold'].mean():.1f}")
col3.metric("Highest daily sales", f"{product_df['UnitsSold'].max():,}")

st.subheader("Historical daily sales")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(
    product_df["Date"],
    product_df["UnitsSold"],
    linewidth=1
)
ax.set_xlabel("Date")
ax.set_ylabel("Units sold")
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Forecast next 7 days recursively
st.subheader("Next 7 days forecast")
st.caption(
    "Forecasts are generated one day at a time using previous actual "
    "sales initially, then the model's own predictions. Results are estimates."
)

history = product_df["UnitsSold"].astype(float).tolist()
last_date = product_df["Date"].max()
forecast_rows = []

for step in range(1, 8):
    next_date = last_date + pd.Timedelta(days=step)

    lag_1 = history[-1]
    lag_7 = history[-7]
    rolling_mean_7 = sum(history[-7:]) / 7

    input_row = pd.DataFrame([{
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_mean_7": rolling_mean_7,
        "DayOfWeek": next_date.dayofweek,
        "Month": next_date.month
    }])[features]

    prediction = max(0, float(model.predict(input_row)[0]))

    forecast_rows.append({
        "Date": next_date,
        "PredictedUnits": round(prediction, 1)
    })

    history.append(prediction)

forecast_df = pd.DataFrame(forecast_rows)

st.dataframe(
    forecast_df,
    use_container_width=True,
    hide_index=True
)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(
    product_df["Date"].tail(30),
    product_df["UnitsSold"].tail(30),
    label="Historical sales"
)
ax.plot(
    forecast_df["Date"],
    forecast_df["PredictedUnits"],
    marker="o",
    label="Forecast"
)
ax.set_xlabel("Date")
ax.set_ylabel("Units sold")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Basic stock planning tool
st.subheader("Inventory planning")

stock = st.number_input(
    "Current stock (units)",
    min_value=0,
    value=100,
    step=10
)

lead_time = st.number_input(
    "Supplier lead time (days)",
    min_value=1,
    max_value=90,
    value=7
)

average_forecast = forecast_df["PredictedUnits"].mean()
expected_demand = average_forecast * lead_time

safety_stock = average_forecast * 2
reorder_point = expected_demand + safety_stock

col1, col2, col3 = st.columns(3)
col1.metric("Expected lead-time demand", f"{expected_demand:.0f} units")
col2.metric("Suggested reorder point", f"{reorder_point:.0f} units")
col3.metric("Current stock", f"{stock:,} units")

if stock <= reorder_point:
    st.warning("Inventory alert: current stock is at or below the suggested reorder point.")
else:
    st.success("Current stock is above the suggested reorder point.")

st.caption(
    "Inventory calculations are illustrative and assume forecast demand "
    "is representative. Validate against real stock, supplier reliability, "
    "and business requirements before making purchasing decisions."
)
