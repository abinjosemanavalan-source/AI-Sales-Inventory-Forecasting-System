
from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "Online Retail.xlsx"

df = pd.read_excel(file_path)

# 1. Inspect cancelled invoices
cancelled = df["InvoiceNo"].astype(str).str.startswith("C")

print("\n--- CANCELLED INVOICE EXAMPLES ---")
print(df.loc[cancelled, [
    "InvoiceNo", "StockCode", "Quantity",
    "UnitPrice", "InvoiceDate"
]].head(10).to_string(index=False))

# 2. Inspect negative quantities
print("\n--- NEGATIVE QUANTITY EXAMPLES ---")
print(df.loc[df["Quantity"] < 0, [
    "InvoiceNo", "StockCode", "Quantity",
    "UnitPrice", "InvoiceDate"
]].head(10).to_string(index=False))

# 3. Inspect zero or negative prices
print("\n--- INVALID PRICE EXAMPLES ---")
print(df.loc[df["UnitPrice"] <= 0, [
    "InvoiceNo", "StockCode", "Quantity",
    "UnitPrice", "InvoiceDate"
]].head(10).to_string(index=False))

# 4. Investigate exact duplicate rows
duplicates = df[df.duplicated(keep=False)]

print("\n--- DUPLICATE EXAMPLES ---")
print(duplicates.sort_values(
    ["InvoiceNo", "StockCode", "InvoiceDate"]
).head(12).to_string(index=False))

# 5. Check how many rows have missing product descriptions
print("\n--- MISSING DESCRIPTIONS ---")
print("Missing descriptions:", df["Description"].isna().sum())

# 6. Check daily coverage
print("\n--- DAILY DATE COVERAGE ---")
print("Unique transaction dates:",
      df["InvoiceDate"].dt.date.nunique())

print("Unique transaction days:",
      df["InvoiceDate"].dt.normalize().nunique())
