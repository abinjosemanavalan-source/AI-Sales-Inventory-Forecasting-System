
from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent
file_path = (
    project_folder
    / "data"
    / "processed"
    / "cleaned_transactions.csv"
)

df = pd.read_csv(
    file_path,
    parse_dates=["InvoiceDate"],
    dtype={"StockCode": str}
)

# Product and date combinations with unusually high sales
events = [
    ("23843", "2011-12-09"),
    ("84077", "2011-10-27"),
    ("22197", "2011-12-08"),
    ("23084", "2011-10-27"),
    ("23084", "2011-11-29"),
]

for product, date in events:
    target_date = pd.Timestamp(date)

    rows = df[
        (df["StockCode"] == product)
        & (df["InvoiceDate"].dt.normalize() == target_date)
    ]

    print(f"\n--- PRODUCT {product} ON {date} ---")

    if rows.empty:
        print("No matching transactions found.")
        continue

    print("Description:", rows["Description"].dropna().unique()[:3])
    print("Transaction lines:", len(rows))
    print("Distinct invoices:", rows["InvoiceNo"].nunique())
    print("Total quantity:", rows["Quantity"].sum())
    print("Total line value:", (rows["Quantity"] * rows["UnitPrice"]).sum())

    print("\nInvoice details:")
    print(
        rows[[
            "InvoiceNo", "Quantity", "UnitPrice", "Country"
        ]].head(10).to_string(index=False)
    )
