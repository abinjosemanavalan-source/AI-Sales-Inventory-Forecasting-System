
from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "Online Retail.xlsx"

df = pd.read_excel(file_path)

print("Original rows:", len(df))

# Remove exact duplicate rows for this initial version
df = df.drop_duplicates().copy()
print("Rows after removing exact duplicates:", len(df))

# Remove cancellation invoices
cancelled = df["InvoiceNo"].astype(str).str.startswith("C")
df = df.loc[~cancelled].copy()
print("Rows after removing cancellations:", len(df))

# Keep positive sales quantities
df = df.loc[df["Quantity"] > 0].copy()
print("Rows after removing non-positive quantities:", len(df))

# Keep positive prices for this first forecasting version
df = df.loc[df["UnitPrice"] > 0].copy()
print("Rows after removing non-positive prices:", len(df))

# Convert each transaction timestamp to its calendar date
df["Date"] = df["InvoiceDate"].dt.normalize()

# Calculate the value of each sales line
df["SalesValue"] = df["Quantity"] * df["UnitPrice"]

# Aggregate transactions into daily product sales
daily_sales = (
    df.groupby(["Date", "StockCode"], as_index=False)
    .agg(
        UnitsSold=("Quantity", "sum"),
        Revenue=("SalesValue", "sum")
    )
)

# Save cleaned transaction data and daily sales data
cleaned_folder = project_folder / "data" / "processed"
cleaned_folder.mkdir(parents=True, exist_ok=True)

df.to_csv(cleaned_folder / "cleaned_transactions.csv", index=False)
daily_sales.to_csv(cleaned_folder / "daily_product_sales.csv", index=False)

print("\n--- CLEANING COMPLETE ---")
print("Cleaned transaction rows:", len(df))
print("Daily product records:", len(daily_sales))
print("Unique products:", daily_sales["StockCode"].nunique())
print("Date range:", daily_sales["Date"].min(), "to", daily_sales["Date"].max())

print("\nSample daily sales:")
print(daily_sales.head(10).to_string(index=False))

print("\nSaved files:")
print("data/processed/cleaned_transactions.csv")
print("data/processed/daily_product_sales.csv")
