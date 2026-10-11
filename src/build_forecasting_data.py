
from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data" / "processed"

sales = pd.read_csv(
    data_folder / "daily_product_sales.csv",
    parse_dates=["Date"],
    dtype={"StockCode": str}
)

products = pd.read_csv(
    data_folder / "top_products.csv",
    dtype={"StockCode": str}
)["StockCode"].tolist()

# Create a continuous calendar
dates = pd.date_range(
    start=sales["Date"].min(),
    end=sales["Date"].max(),
    freq="D"
)

# Create every date-product combination
calendar = pd.MultiIndex.from_product(
    [dates, products],
    names=["Date", "StockCode"]
).to_frame(index=False)

# Add the recorded sales
forecast = calendar.merge(
    sales[sales["StockCode"].isin(products)],
    on=["Date", "StockCode"],
    how="left"
)

# Missing product-days have no recorded positive sales
forecast["UnitsSold"] = forecast["UnitsSold"].fillna(0)
forecast["Revenue"] = forecast["Revenue"].fillna(0)

forecast["UnitsSold"] = forecast["UnitsSold"].astype(int)

# Sort the dataset for time-series modelling
forecast = forecast.sort_values(
    ["StockCode", "Date"]
).reset_index(drop=True)

output_path = data_folder / "forecasting_dataset.csv"
forecast.to_csv(output_path, index=False)

print("--- FORECASTING DATASET ---")
print("Total rows:", len(forecast))
print("Unique products:", forecast["StockCode"].nunique())
print("Days per product:", len(dates))
print("Date range:", dates.min(), "to", dates.max())

print("\nSample records:")
print(forecast.head(10).to_string(index=False))

print("\nSales summary by product:")
print(
    forecast.groupby("StockCode")["UnitsSold"]
    .agg(["sum", "mean", "max"])
    .rename(columns={
        "sum": "TotalUnits",
        "mean": "AverageDailyUnits",
        "max": "MaxDailyUnits"
    })
    .round(2)
    .to_string()
)

print("\nSaved:", output_path)
