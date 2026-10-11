
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data" / "processed"

df = pd.read_csv(data_folder / "backtest_results.csv")
df["Date"] = pd.to_datetime(df["Date"])

df["ModelError"] = (
    df["UnitsSold"] - df["PredictedUnits"]
).abs()

df["BaselineError"] = (
    df["UnitsSold"] - df["BaselinePrediction"]
).abs()

chosen_rows = []

for week in sorted(df["TestWeek"].unique()):
    current = df[df["TestWeek"] == week].copy()

    # Only previous weeks can guide this week's model choice
    previous = df[df["TestWeek"] < week]

    for _, row in current.iterrows():
        product = row["StockCode"]

        if previous.empty:
            # No past backtest results yet: use the baseline
            choice = "Baseline"
        else:
            history = previous[previous["StockCode"] == product]

            model_mae = history["ModelError"].mean()
            baseline_mae = history["BaselineError"].mean()

            choice = (
                "Random Forest"
                if model_mae < baseline_mae
                else "Baseline"
            )

        prediction = (
            row["PredictedUnits"]
            if choice == "Random Forest"
            else row["BaselinePrediction"]
        )

        record = row.to_dict()
        record["ChosenModel"] = choice
        record["HybridPrediction"] = prediction
        chosen_rows.append(record)

hybrid = pd.DataFrame(chosen_rows)

def report(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

print("--- HYBRID STRATEGY BACKTEST ---")
print("Test records:", len(hybrid))

report(
    "Random Forest",
    hybrid["UnitsSold"],
    hybrid["PredictedUnits"]
)

report(
    "Seven-day baseline",
    hybrid["UnitsSold"],
    hybrid["BaselinePrediction"]
)

report(
    "Hybrid strategy",
    hybrid["UnitsSold"],
    hybrid["HybridPrediction"]
)

print("\nModel choices across all test weeks:")
print(hybrid["ChosenModel"].value_counts().to_string())

output_path = data_folder / "hybrid_backtest_results.csv"
hybrid.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)
