
from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "Online Retail.xlsx"

df = pd.read_excel(file_path)

# 1. Check the date range
print("\n--- DATE RANGE ---")
print("First date:", df["InvoiceDate"].min())
print("Last date:", df["InvoiceDate"].max())

# 2. Count different products
print("\n--- PRODUCTS ---")
print("Unique products:", df["StockCode"].nunique())

# 3. Investigate cancellations
cancelled = df["InvoiceNo"].astype(str).str.startswith("C")
print("\n--- CANCELLATIONS ---")
print("Cancellation invoice lines:", cancelled.sum())

# 4. Investigate quantities
print("\n--- QUANTITIES ---")
print("Negative quantity rows:", (df["Quantity"] < 0).sum())
print("Zero quantity rows:", (df["Quantity"] == 0).sum())

# 5. Investigate prices
print("\n--- PRICES ---")
print("Zero or negative price rows:", (df["UnitPrice"] <= 0).sum())

# 6. Check duplicate rows
print("\n--- DUPLICATES ---")
print("Duplicate rows:", df.duplicated().sum())

# 7. Inspect the countries
print("\n--- TOP 10 COUNTRIES ---")
print(df["Country"].value_counts().head(10))

# 8. Calculate recorded transaction value
df["LineValue"] = df["Quantity"] * df["UnitPrice"]

print("\n--- TRANSACTION VALUE ---")
print(df["LineValue"].describe())
