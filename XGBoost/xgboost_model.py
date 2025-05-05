import os
import getpass
import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from scipy.stats import uniform, randint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
from xgboost import XGBRegressor

# XGBoost_Forecaster_Rolling
def XGBoost_Forecaster(X, y, forecast_horizon, last_observation_date, scaler, window_length=108, verbose=True):
    """
    Forecast inflation using one XGBoost model per forecast horizon (direct forecast), based on Garcia et al. (2017)
    
    Args:
        X: DataFrame of predictors
        y: Series of target variable
        forecast_horizon: int, number of months ahead to forecast
        last_observation_date: str or Timestamp, last available observation used for forecasting
        scaler: fitted sklearn scaler for transforming X
        window_length: int, number of months in the rolling window (default: 108)
        verbose: bool, whether to print model training details
    """

    # --- STEP 1: Restrict data to what's available up to the forecast date ---
    X = X.loc[:last_observation_date]
    y = y.loc[:last_observation_date]

    # Ensure we have enough data for the rolling window
    if len(X) < window_length:
        raise ValueError("Not enough data for the rolling window.")

    # --- STEP 2: Extract rolling window ending at last_observation_date ---
    X_window = X.iloc[-window_length:]
    y_window = y.iloc[-window_length:]

    # Dictionary to store one trained model per horizon
    xgb_models = {}

    # --- STEP 3: Train one model for each forecast horizon ---
    for h in range(forecast_horizon):
        if verbose: 
            print(f"\n=== Horizon h={h+1} ===")
        
        # Shift target variable to align with current horizon
        y_shifted = y_window.shift(-h).dropna()
        X_train = X_window.iloc[:len(y_shifted)]  # Align X and y
        y_train = y_shifted

        # Scale the predictor variables
        X_train_scaled = scaler.transform(X_train)

        # Define and train the XGBoost model
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=235,
            learning_rate=0.306066,
            max_depth=3,
            subsample=0.6022,
            colsample_bytree=0.6298,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)
        xgb_models[h] = model

        if verbose:
            print(f"Number of training observations: {len(y_train)}")
            print(f"Number of features: {X_train.shape[1]}")

    # --- STEP 4: Generate forecasts from the most recent observation ---
    X_t = X.loc[[last_observation_date]]           # The row used for forecasting
    X_t_scaled = scaler.transform(X_t)

    xgb_forecasts = {}
    for h in range(forecast_horizon):
        forecast = xgb_models[h].predict(X_t_scaled)
        xgb_forecasts[h] = forecast[0]             # Store scalar forecast

    # --- STEP 5: Format output into a tidy DataFrame ---
    start_date = pd.to_datetime(last_observation_date) + pd.DateOffset(months=1)
    forecast_dates = [start_date + pd.DateOffset(months=h) for h in xgb_forecasts.keys()]

    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Inflation forecast": list(xgb_forecasts.values()),
        "Horizon": list(xgb_forecasts.keys())
    })

    if verbose:
        print("\nForecasted months:")
        for date in forecast_dates:
            print(date.strftime("%Y-%m"))

    return forecast_df

def run_rolling_forecast(X, y, forecast_horizon=12, start_date="2012-01", end_date="2015-12", window_length=72):
    """
    Runs a rolling real-time forecast using XGBoost.
    One forecast is made each month, with predictions for the next `forecast_horizon` periods.

    Args:
        X (pd.DataFrame): Predictor variables.
        y (pd.Series): Target variable.
        forecast_horizon (int): Number of months ahead to forecast.
        start_date (str): Start date for forecasting (format "YYYY-MM").
        end_date (str): End date for forecasting.
        window_length (int): Number of most recent observations used for training.

    Returns:
        pd.DataFrame: Concatenated forecast results for each origin date.
    """

    all_forecasts = []

    # Generate list of monthly forecast origin dates
    forecast_dates = pd.date_range(start=start_date, end=end_date, freq="MS")

    for date in forecast_dates:
        # Extract training data available up to the current forecast date (real-time simulation)
        X_train = X.loc[:date]
        y_train = y.loc[:date]

        # Skip this iteration if not enough data in the rolling window
        if len(X_train) < window_length:
            print("Skipping due to insufficient data.")
            continue

        # Standardize predictors using data from the rolling window
        scaler = StandardScaler()
        scaler.fit(X_train.iloc[-window_length:])

        # Forecast the next `forecast_horizon` months using XGBoost
        forecast_df = XGBoost_Forecaster(
            X=X,
            y=y,
            forecast_horizon=forecast_horizon,
            last_observation_date=date,
            scaler=scaler,
            window_length=window_length,
            verbose=False
        )

        # Record the forecast origin
        forecast_df["Forecast_made_in"] = date
        all_forecasts.append(forecast_df)

    # Combine all individual forecast results into a single DataFrame
    all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    return all_forecasts_df



def evaluate_forecasts(forecast_df, y):
    """
    Matches forecasts with actual values and returns RMSE and MAE by forecast horizon.

    Args:
        forecast_df (pd.DataFrame): DataFrame containing forecast results with columns "Date", "Inflation forecast", and "Horizon".
        y (pd.Series): Actual observed values (e.g., true inflation), indexed by date.

    Returns:
        evaluation (pd.DataFrame): Aggregated evaluation metrics (MAE, RMSE, count) per horizon.
        merged (pd.DataFrame): Full merged DataFrame including forecast errors.
    """

    # Merge forecasts with actual observed values
    merged = forecast_df.merge(
        y.rename("y_true"), 
        left_on="Date", 
        right_index=True,
        how="left"
    )

    # Drop future periods where the actual value is not yet available
    merged = merged.dropna(subset=["y_true"])

    # Compute forecast errors
    merged["error"] = merged["Inflation forecast"] - merged["y_true"]
    merged["abs_error"] = merged["error"].abs()
    merged["squared_error"] = merged["error"] ** 2

    # Aggregate error metrics by forecast horizon
    evaluation = merged.groupby("Horizon").agg(
        MAE=("abs_error", "mean"),
        RMSE=("squared_error", lambda x: (x.mean())**0.5),
        N_obs=("y_true", "count")
    ).reset_index()

    return evaluation, merged
