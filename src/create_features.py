
from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data" / "processed"

file_path = data_folder / "forecasting_dataset.csv"

df = pd.read_csv(
    file_path,
    parse_dates=["Date"],
    dtype={"StockCode": str}
)

df = df.sort_values(
    ["StockCode", "Date"]
).reset_index(drop=True)

# Calendar features
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month

# Previous sales for each product
df["lag_1"] = df.groupby("StockCode")["UnitsSold"].shift(1)
df["lag_7"] = df.groupby("StockCode")["UnitsSold"].shift(7)

# Average sales over the previous seven days
df["rolling_mean_7"] = (
    df.groupby("StockCode")["UnitsSold"]
    .transform(lambda sales: sales.shift(1).rolling(7).mean())
)

# Remove rows that do not have enough historical data
df = df.dropna().reset_index(drop=True)

output_path = data_folder / "features_dataset.csv"
df.to_csv(output_path, index=False)

print("--- FEATURE DATASET ---")
print("Rows:", len(df))
print("Products:", df["StockCode"].nunique())

print("\nFeature columns:")
print(df.columns.tolist())

print("\nSample features:")
print(
    df[[
        "Date", "StockCode", "UnitsSold",
        "lag_1", "lag_7", "rolling_mean_7",
        "DayOfWeek", "Month"
    ]].head(10).to_string(index=False)
)

print("\nSaved:", output_path)
