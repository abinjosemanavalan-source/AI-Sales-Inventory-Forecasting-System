
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project paths
project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data" / "processed"

# Load datasets
sales = pd.read_csv(
    data_folder / "forecasting_dataset.csv",
    parse_dates=["Date"],
    dtype={"StockCode": str}
)

features_df = pd.read_csv(
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

# Test the final 56 days in 8 non-overlapping weekly periods
last_date = sales["Date"].max()
first_test_start = last_date - pd.DateOffset(days=55)

# Prepare the seven-day baseline:
# prediction for date D = actual sales on date D - 7 days
baseline = sales[["Date", "StockCode", "UnitsSold"]].copy()
baseline["Date"] = baseline["Date"] + pd.DateOffset(days=7)
baseline = baseline.rename(
    columns={"UnitsSold": "BaselinePrediction"}
)

all_results = []

for week in range(8):
    test_start = first_test_start + pd.DateOffset(days=7 * week)
    test_end = test_start + pd.DateOffset(days=6)

    # Only use training rows available before this test week
    train = features_df[
        features_df["Date"] < test_start
    ].copy()

    test = sales[
        (sales["Date"] >= test_start)
        & (sales["Date"] <= test_end)
    ].copy()

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(train[features], train["UnitsSold"])

    predictions = []

    # Predict each product's seven days recursively
    for product in sorted(sales["StockCode"].unique()):
        product_history = sales[
            (sales["StockCode"] == product)
            & (sales["Date"] < test_start)
        ].sort_values("Date")["UnitsSold"].astype(float).tolist()

        for day_offset in range(7):
            forecast_date = test_start + pd.DateOffset(
                days=day_offset
            )

            row = pd.DataFrame([{
                "lag_1": product_history[-1],
                "lag_7": product_history[-7],
                "rolling_mean_7": np.mean(product_history[-7:]),
                "DayOfWeek": forecast_date.dayofweek,
                "Month": forecast_date.month
            }])[features]

            prediction = max(
                0.0,
                float(model.predict(row)[0])
            )

            predictions.append({
                "Date": forecast_date,
                "StockCode": product,
                "PredictedUnits": prediction
            })

            # Use the prediction, not actual future sales
            product_history.append(prediction)

    predictions_df = pd.DataFrame(predictions)

    results = test.merge(
        predictions_df,
        on=["Date", "StockCode"],
        how="inner"
    )

    results = results.merge(
        baseline,
        on=["Date", "StockCode"],
        how="left"
    )

    results["TestWeek"] = week + 1
    all_results.append(results)

# Combine all eight test weeks
final_results = pd.concat(all_results, ignore_index=True)

def report_metrics(name, data):
    print(f"\n{name} ({len(data)} records)")

    for model_name, column in [
        ("Random Forest", "PredictedUnits"),
        ("Seven-day baseline", "BaselinePrediction")
    ]:
        mae = mean_absolute_error(
            data["UnitsSold"], data[column]
        )
        rmse = mean_squared_error(
            data["UnitsSold"], data[column
            ]
        ) ** 0.5

        print(f"  {model_name}:")
        print(f"    MAE:  {mae:.2f}")
        print(f"    RMSE: {rmse:.2f}")

print("--- ROLLING 7-DAY BACKTEST ---")
print("Test period:", first_test_start.date(), "to", last_date.date())
print("Weekly test periods:", final_results["TestWeek"].nunique())
print("Total test records:", len(final_results))

for week in sorted(final_results["TestWeek"].unique()):
    week_data = final_results[
        final_results["TestWeek"] == week
    ]

    print(
        f"\nWeek {week}: "
        f"{week_data['Date'].min().date()} to "
        f"{week_data['Date'].max().date()}"
    )

    report_metrics("Weekly results", week_data)

# Overall performance across all test weeks
report_metrics("OVERALL RESULTS", final_results)

# Diagnostic comparison; does not remove any records
normal = final_results[final_results["UnitsSold"] < 1000]
high_sales = final_results[final_results["UnitsSold"] >= 1000]

report_metrics("Normal sales (< 1000 units)", normal)
report_metrics("High sales (>= 1000 units)", high_sales)

output_path = data_folder / "backtest_results.csv"
final_results.to_csv(output_path, index=False)

print("\nDetailed results saved to:", output_path)
