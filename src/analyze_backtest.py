
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
file_path = ROOT / "data" / "processed" / "backtest_results.csv"

df = pd.read_csv(file_path)
df["Date"] = pd.to_datetime(df["Date"])

# Calculate absolute prediction errors
df["ModelError"] = (df["UnitsSold"] - df["PredictedUnits"]).abs()
df["BaselineError"] = (
    df["UnitsSold"] - df["BaselinePrediction"]
).abs()

# Identify unusually large actual sales for diagnosis only
spike_threshold = 1000
spikes = df[df["UnitsSold"] >= spike_threshold]
normal = df[df["UnitsSold"] < spike_threshold]

def show_metrics(name, data):
    if data.empty:
        print(f"\n{name}: No records")
        return

    model_mae = data["ModelError"].mean()
    baseline_mae = data["BaselineError"].mean()

    model_rmse = np.sqrt(
        ((data["UnitsSold"] - data["PredictedUnits"]) ** 2).mean()
    )

    print(f"\n{name} ({len(data)} records)")
    print(f"Random Forest MAE: {model_mae:.2f}")
    print(f"Baseline MAE:      {baseline_mae:.2f}")
    print(f"Random Forest RMSE: {model_rmse:.2f}")

print("--- BACKTEST ERROR ANALYSIS ---")

show_metrics("All test records", df)
show_metrics(f"Normal sales (< {spike_threshold} units)", normal)
show_metrics(f"High sales (>= {spike_threshold} units)", spikes)

print("\n--- HIGH-SALES RECORDS ---")
if spikes.empty:
    print("No high-sales records found.")
else:
    print(
        spikes[
            [
                "Date",
                "StockCode",
                "UnitsSold",
                "PredictedUnits",
                "BaselinePrediction",
                "ModelError",
                "BaselineError",
            ]
        ].sort_values("UnitsSold", ascending=False).to_string(index=False)
    )
.