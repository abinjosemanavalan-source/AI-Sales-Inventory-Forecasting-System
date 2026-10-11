
from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data" / "processed"
model_folder = project_folder / "models"
model_folder.mkdir(exist_ok=True)

df = pd.read_csv(
    data_folder / "features_dataset.csv",
    parse_dates=["Date"],
    dtype={"StockCode": str}
)

features = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "DayOfWeek",
    "Month"
]

X = df[features]
y = df["UnitsSold"]

# Train on all available historical records
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

model_path = model_folder / "sales_forecast_model_final.pkl"

joblib.dump(
    {
        "model": model,
        "features": features
    },
    model_path
)

print("Final model trained successfully!")
print("Training records:", len(df))
print("Products:", df["StockCode"].nunique())
print("Model saved to:", model_path)
