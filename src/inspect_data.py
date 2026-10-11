
from pathlib import Path
import pandas as pd

# Find the project folder
project_folder = Path(__file__).resolve().parent.parent

# Find the Excel dataset
file_path = project_folder / "data" / "Online Retail.xlsx"

# Check whether the file exists
if not file_path.exists():
    print("Dataset not found!")
    print("Check the file name and data folder.")
else:
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Display basic information
    print("Dataset loaded successfully!")
    print("Shape:", df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData information:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nQuantity summary:")
    print(df["Quantity"].describe())
