
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data" / "processed"

df = pd.read_csv(
    data_folder / "features_dataset.csv",
    parse_dates=["Date"],
    dtype={"StockCode": str}
)

# Keep the same chronological test period
split_date = df["Date"].quantile(0.8)
train = df[df["Date"] < split_date].copy()
test = df[df["Date"] >= split_date].copy()

saved = joblib.load(
    project_folder / "models" / "sales_forecast_model.pkl"
)

features = saved["features"]

X_train = train[features]
y_train = train["UnitsSold"]
X_test = test[features]
y_test = test["UnitsSold"]

# Model 1: Existing Random Forest
original_model = saved["model"]
original_predictions = original_model.predict(X_test)

# Model 2: Train Random Forest on log-transformed sales
log_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

log_model.fit(X_train, np.log1p(y_train))

log_predictions = np.expm1(log_model.predict(X_test))
log_predictions = np.maximum(log_predictions, 0)

# Model 3: Seven-day baseline
baseline_predictions = test["lag_7"].to_numpy()

# Compare all three on the same test records
actual = y_test.to_numpy()

models = {
    "Seven-day baseline": baseline_predictions,
    "Original Random Forest": original_predictions,
    "Log-transformed Random Forest": log_predictions
}

print("--- MODEL COMPARISON ---")

for name, predictions in models.items():
    mae = mean_absolute_error(actual, predictions)
    rmse = mean_squared_error(actual, predictions) ** 0.5

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

# Check the results without removing any sales events
results = test[["Date", "StockCode", "UnitsSold"]].copy()
results["LogModelPrediction"] = np.round(log_predictions, 2)
results["AbsoluteError"] = np.abs(
    results["UnitsSold"] - results["LogModelPrediction"]
)

print("\n--- LARGEST ERRORS FOR LOG MODEL ---")
print(
    results.nlargest(5, "AbsoluteError")
    .to_string(index=False)
)

print("\nComparison complete. Original data and model remain unchanged.")
