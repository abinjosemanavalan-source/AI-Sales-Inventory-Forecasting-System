
from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "processed" / "daily_product_sales.csv"

df = pd.read_csv(file_path, parse_dates=["Date"])

print("Dataset shape:", df.shape)
print("Unique products:", df["StockCode"].nunique())
print("Date range:", df["Date"].min(), "to", df["Date"].max())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTop 10 products by units sold:")
print(
    df.groupby("StockCode")["UnitsSold"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nDaily sales summary:")
print(df["UnitsSold"].describe())

print("\nSample records:")
print(df.head(10).to_string(index=False))
