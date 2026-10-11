
from pathlib import Path
import pandas as pd
import numpy as np

project_folder = Path(__file__).resolve().parent.parent
file_path = (
    project_folder
    / "data"
    / "processed"
    / "backtest_results.csv"
)

df = pd.read_csv(file_path)

df["ModelError"] = (
    df["UnitsSold"] - df["PredictedUnits"]
).abs()

df["BaselineError"] = (
    df["UnitsSold"] - df["BaselinePrediction"]
).abs()

report = df.groupby("StockCode").agg(
    TestDays=("UnitsSold", "count"),
    TotalUnits=("UnitsSold", "sum"),
    ModelMAE=("ModelError", "mean"),
    BaselineMAE=("BaselineError", "mean")
).reset_index()

report["ImprovementPercent"] = (
    (report["BaselineMAE"] - report["ModelMAE"])
    / report["BaselineMAE"].replace(0, np.nan)
    * 100
)

report = report.sort_values(
    "ImprovementPercent",
    ascending=False
)

print("--- PRODUCT-LEVEL BACKTEST REPORT ---")
print(
    report.round(2).to_string(index=False)
)

output_path = (
    project_folder
    / "data"
    / "processed"
    / "product_backtest_report.csv"
)

report.to_csv(output_path, index=False)
print("\nReport saved to:", output_path)
