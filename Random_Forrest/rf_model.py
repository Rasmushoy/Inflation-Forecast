import os
import getpass
import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# RandomForest_Forecaster_Rolling
def RandomForest_Forecaster(X, y, forecast_horizon, last_observation_date, scaler, trees = 200, window_length=108, verbose=True):
    """
    Forecast inflation using one Random Forest model per forecast horizon (direct forecast),
    based on the method in Garcia et al. (2017).
    
    Args:
        X: DataFrame of predictors
        y: Series of target variable
        forecast_horizon: int, how many steps ahead to forecast (e.g., 12)
        last_observation_date: str or Timestamp, the forecast origin
        scaler: fitted sklearn scaler (to prevent data leakage)
        trees: int, number of trees in each Random Forest
        window_length: int, number of time steps in rolling window (default: 108 months)
        verbose: bool, whether to print training and forecasting details
    """

    # --- STEP 1: Restrict to real-time available data ---
    X = X.loc[:last_observation_date]
    y = y.loc[:last_observation_date]

    # --- STEP 2: Apply rolling window ---
    if len(X) < window_length:
        raise ValueError("Not enough data for the chosen rolling window length.")
    
    X_window = X.iloc[-window_length:]
    y_window = y.iloc[-window_length:]

    # Dictionary to hold one model per forecast horizon
    rf_models = {}

    # --- STEP 3: Train one Random Forest per horizon ---
    for h in range(forecast_horizon):
        if verbose: 
            print(f"\n=== Horizon h={h+1} ===")

        # Shift target to create y_{t+h}
        y_shifted = y_window.shift(-h).dropna()

        # Align predictors with shifted target
        X_train = X_window.iloc[:len(y_shifted)]
        y_train = y_shifted

        # Train model
        model = RandomForestRegressor(n_estimators=trees, random_state=42)
        model.fit(scaler.transform(X_train), y_train)

        # Store trained model
        rf_models[h] = model

        if verbose: 
            print(f"Number of training observations: {len(y_train)}")
            print(f"Number of input features: {model.n_features_in_}")

    # --- STEP 4: Forecast using the most recent observation ---
    X_t = X.loc[[last_observation_date]]  # Real-time input row
    X_t_scaled = scaler.transform(X_t)

    rf_forecasts = {}
    for h in range(forecast_horizon):
        forecast = rf_models[h].predict(X_t_scaled)
        rf_forecasts[h] = forecast[0]  # Store forecast as float

    # --- STEP 5: Generate forecast dates ---
    start_date = pd.to_datetime(last_observation_date) + pd.DateOffset(months=1)
    forecast_dates = [start_date + pd.DateOffset(months=h) for h in rf_forecasts.keys()]

    if verbose: 
        print("\nForecasted months:")
        for date in forecast_dates:
            print(date.strftime("%Y-%m"))

    # --- STEP 6: Format output as DataFrame ---
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Inflation forecast": list(rf_forecasts.values()),
        "Horizon": list(rf_forecasts.keys())
    })

    return forecast_df



def run_rolling_forecast(X, y, forecast_horizon=12, start_date="2012-01", end_date="2015-12", window_length=72, Trees=200):
    """
    Runs a rolling real-time forecast using Random Forest.
    One forecast is made per month, with predictions for the next `forecast_horizon` periods.

    Args:
        X (pd.DataFrame): Predictor variables.
        y (pd.Series): Target variable.
        forecast_horizon (int): Number of months ahead to forecast.
        start_date (str): Start of the forecasting window (format: "YYYY-MM").
        end_date (str): End of the forecasting window.
        window_length (int): Number of observations used in each rolling window.
        Trees (int): Number of trees in the Random Forest.

    Returns:
        pd.DataFrame: All forecasts with metadata on when each forecast was made.
    """

    all_forecasts = []

    # Generate list of monthly forecast origin dates
    forecast_dates = pd.date_range(start=start_date, end=end_date, freq="MS")

    for date in forecast_dates:
        # Select real-time training data available at the forecast date
        X_train = X.loc[:date]
        y_train = y.loc[:date]

        # Skip if not enough data in the rolling window
        if len(X_train) < window_length:
            print("Skipping due to insufficient data.")
            continue

        # Fit a scaler on the latest `window_length` observations
        scaler = StandardScaler()
        scaler.fit(X_train.iloc[-window_length:])

        # Generate forecasts for all horizons using Random Forest
        forecast_df = RandomForest_Forecaster(
            X=X,
            y=y,
            forecast_horizon=forecast_horizon,
            last_observation_date=date,
            scaler=scaler,
            trees=Trees,
            window_length=window_length,
            verbose=False
        )

        # Record when the forecast was made
        forecast_df["Forecast_made_in"] = date
        all_forecasts.append(forecast_df)

    # Combine all individual forecast DataFrames into a single result
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
