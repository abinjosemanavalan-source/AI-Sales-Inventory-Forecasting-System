
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data" / "processed"
model_path = project_folder / "models" / "sales_forecast_model.pkl"

df = pd.read_csv(
    data_folder / "features_dataset.csv",
    parse_dates=["Date"],
    dtype={"StockCode": str}
)

saved = joblib.load(model_path)
model = saved["model"]
features = saved["features"]

# Use the same chronological split as training
split_date = df["Date"].quantile(0.8)
test = df[df["Date"] >= split_date].copy()

test["PredictedUnits"] = model.predict(test[features])
test["AbsoluteError"] = (
    test["UnitsSold"] - test["PredictedUnits"]
).abs()

# Overall metrics
print("--- OVERALL TEST RESULTS ---")
print("MAE:", round(test["AbsoluteError"].mean(), 2))

rmse = mean_squared_error(
    test["UnitsSold"], test["PredictedUnits"]
) ** 0.5

print("RMSE:", round(rmse, 2))

# Evaluate each product separately
print("\n--- ERROR BY PRODUCT ---")

product_errors = test.groupby("StockCode").apply(
    lambda group: pd.Series({
        "MAE": mean_absolute_error(
            group["UnitsSold"], group["PredictedUnits"]
        ),
        "RMSE": mean_squared_error(
            group["UnitsSold"], group["PredictedUnits"]
        ) ** 0.5,
        "MaxError": group["AbsoluteError"].max()
    }),
    include_groups=False
)

print(product_errors.round(2).sort_values("MAE", ascending=False))

# Inspect the five largest individual errors
print("\n--- FIVE LARGEST PREDICTION ERRORS ---")

worst = test.nlargest(5, "AbsoluteError")

print(
    worst[[
        "Date", "StockCode", "UnitsSold",
        "PredictedUnits", "AbsoluteError"
    ]].to_string(index=False)
)
