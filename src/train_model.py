
from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.dummy import DummyRegressor

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data" / "processed"
model_folder = project_folder / "models"
model_folder.mkdir(exist_ok=True)

df = pd.read_csv(
    data_folder / "features_dataset.csv",
    parse_dates=["Date"],
    dtype={"StockCode": str}
)

# Use earlier dates for training and later dates for testing
split_date = df["Date"].quantile(0.8)

train = df[df["Date"] < split_date].copy()
test = df[df["Date"] >= split_date].copy()

features = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "DayOfWeek",
    "Month"
]

X_train = train[features]
y_train = train["UnitsSold"]

X_test = test[features]
y_test = test["UnitsSold"]

# Baseline: predict the same sales as seven days ago
baseline_predictions = test["lag_7"]

# Train the Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Calculate evaluation metrics
baseline_mae = mean_absolute_error(y_test, baseline_predictions)
model_mae = mean_absolute_error(y_test, predictions)

baseline_rmse = mean_squared_error(
    y_test, baseline_predictions
) ** 0.5

model_rmse = mean_squared_error(
    y_test, predictions
) ** 0.5

print("--- DATA SPLIT ---")
print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Training ends:", train["Date"].max())
print("Testing starts:", test["Date"].min())

print("\n--- MODEL RESULTS ---")
print(f"Baseline MAE: {baseline_mae:.2f}")
print(f"Random Forest MAE: {model_mae:.2f}")
print(f"Baseline RMSE: {baseline_rmse:.2f}")
print(f"Random Forest RMSE: {model_rmse:.2f}")

print("\n--- SAMPLE PREDICTIONS ---")
results = test[["Date", "StockCode", "UnitsSold"]].copy()
results["PredictedUnits"] = predictions.round(2)

print(results.head(15).to_string(index=False))

# Save the trained model
joblib.dump(
    {"model": model, "features": features},
    model_folder / "sales_forecast_model.pkl"
)

print("\nModel saved to models/sales_forecast_model.pkl")
