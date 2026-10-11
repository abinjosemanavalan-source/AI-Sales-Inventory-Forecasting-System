# AI Sales & Inventory Forecasting System

A machine learning project that analyzes historical retail transactions, forecasts product demand, and provides inventory-planning insights through an interactive Streamlit dashboard.

## Project Overview

Retail businesses need to estimate future demand to manage inventory efficiently. This project uses historical retail transaction data to build a demand forecasting model and demonstrate how predictions can support inventory planning.

The system includes data cleaning, exploratory analysis, feature engineering, machine learning, chronological evaluation, rolling backtesting, and an interactive dashboard.

> **Note:** The project uses historical retail data from 2010–2011. Forecasts demonstrate the modeling workflow and are not real-time predictions of current demand.

## Objectives

- Clean and prepare retail transaction data.
- Identify high-volume products for forecasting.
- Build product-level daily sales datasets.
- Engineer time-based and lag-based features.
- Train and evaluate a Random Forest regression model.
- Compare predictions against a seven-day seasonal-naive baseline.
- Evaluate forecasting performance using rolling seven-day backtesting.
- Visualize historical sales, future forecasts, and illustrative inventory recommendations.

## Technology Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Model persistence:** Joblib
- **Dashboard:** Streamlit
- **Visualization:** Streamlit charts
- **Development environment:** Visual Studio Code

## Dataset

**Source:** UCI Machine Learning Repository — Online Retail

The dataset contains transactions from a UK-based online retailer between December 2010 and December 2011.

Dataset link: https://archive.ics.uci.edu/dataset/352/online+retail

Original dataset: 541,909 transaction records across eight columns, including invoice number, product code, description, quantity, invoice date, unit price, customer ID, and country.

The dataset is licensed under **CC BY 4.0**. Refer to the original dataset page for attribution and license details.

## Project Workflow

1. **Data inspection:** Examine dataset structure, missing values, duplicates, cancellations, and unusual quantities.
2. **Data cleaning:** Remove exact duplicate rows, cancellation invoices, non-positive quantities, and non-positive unit prices.
3. **Product selection:** Select the ten products with the highest total recorded sales quantity.
4. **Daily aggregation:** Aggregate transactions into daily product-level sales records.
5. **Calendar completion:** Represent missing product-date combinations with zero recorded qualifying sales.
6. **Feature engineering:** Create lag features, rolling averages, day-of-week, and month features.
7. **Model training:** Train a Random Forest regressor using a chronological train/test split.
8. **Evaluation:** Compare the model with a seven-day seasonal-naive baseline.
9. **Rolling backtesting:** Evaluate repeated seven-day forecasting periods.
10. **Dashboard:** Display historical sales, recursive seven-day forecasts, and illustrative inventory-planning calculations.

## Features Used

| Feature | Description |
|---|---|
| `lag_1` | Product sales recorded one day earlier |
| `lag_7` | Product sales recorded seven days earlier |
| `rolling_mean_7` | Mean sales across the previous seven days, excluding the prediction day |
| `DayOfWeek` | Day of the week |
| `Month` | Calendar month |

The rolling mean is shifted to avoid using the target day's sales when calculating its features.

## Models

### Random Forest Regressor

The primary model is a Scikit-learn `RandomForestRegressor` with:

- 100 decision trees
- Maximum tree depth of 12
- Random state of 42

### Baseline Model

The baseline predicts sales using the corresponding value from seven days earlier. This provides a simple benchmark for weekly retail demand patterns.

## Model Evaluation

### Chronological holdout test

The initial evaluation used a chronological split rather than randomly shuffling dates.

| Metric | Seven-day baseline | Random Forest |
|---|---:|---:|
| Mean Absolute Error (MAE) | 258.47 | **233.22** |
| Root Mean Squared Error (RMSE) | 3,006.13 | **2,996.47** |

The Random Forest reduced MAE compared