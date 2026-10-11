
from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent

file_path = (
    project_folder
    / "data"
    / "processed"
    / "daily_product_sales.csv"
)

df = pd.read_csv(file_path, parse_dates=["Date"])

# Total sales for each product
product_summary = (
    df.groupby("StockCode")
    .agg(
        TotalUnits=("UnitsSold", "sum"),
        TotalRevenue=("Revenue", "sum"),
        ActiveDays=("Date", "nunique")
    )
    .sort_values("TotalUnits", ascending=False)
)

print("--- TOP 15 PRODUCTS BY UNITS SOLD ---")
print(product_summary.head(15).to_string())

# Select the top 10 products for our first model
top_products = product_summary.head(10).index.tolist()

print("\nSelected product codes:")
print(top_products)

# Save the selected products
output_path = project_folder / "data" / "processed" / "top_products.csv"

product_summary.head(10).reset_index().to_csv(
    output_path, index=False
)

print("\nSaved:", output_path)
