import os
import getpass
import numpy as np
import pandas as pd
from time import time
from functools import reduce
import matplotlib.pyplot as plt

# Sklearn
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def pcr_forecaster(X, y, forecast_horizon, last_observation_date, scaler,
                   variance_threshold=0.95, window_length=108, verbose=True):
    """
    Forecasts inflation using Principal Component Regression (PCR), one model per forecast horizon.

    Args:
        X (pd.DataFrame): Predictor variables.
        y (pd.Series): Target variable.
        forecast_horizon (int): Number of months ahead to forecast.
        last_observation_date (str or pd.Timestamp): Date from which the forecast is made.
        scaler (StandardScaler): Pre-fitted scaler used to transform data.
        variance_threshold (float): Cumulative explained variance to retain (default: 0.95).
        window_length (int): Number of observations used for training.
        verbose (bool): Whether to print progress messages.

    Returns:
        pd.DataFrame: Forecast results with date, predicted value, and horizon.
    """

    # Trim data to simulate real-time availability
    X_init = X.loc[:last_observation_date]
    y_init = y.loc[:last_observation_date]
    
    # Use only the most recent `window_length` observations
    X = X_init.iloc[-window_length:]
    y = y_init.iloc[-window_length:]

    # Standardize the predictors
    X_scaled = scaler.transform(X)

    pcr_models = {}

    # Fit a separate PCR model for each forecast horizon
    for h in range(forecast_horizon):
        if verbose:
            print(f"\n=== Horizon h={h} ===")

        # Shift target forward to match forecast horizon
        y_shifted = y.shift(-h).dropna()
        X_train = X_scaled[:len(y_shifted)]
        y_train = y_shifted

        # Automatically select number of components to retain desired explained variance
        pca = PCA().fit(X_train)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumvar >= variance_threshold) + 1

        if verbose:
            print(f"Explained variance (k={n_components}): {cumvar[n_components - 1]:.2%}")
        
        # Define PCR model as a pipeline
        model = Pipeline([
            ("pca", PCA(n_components=n_components)),
            ("reg", LinearRegression())
        ])

        model.fit(X_train, y_train)
        pcr_models[h] = model

        if verbose:
            print(f"Training obs: {len(y_train)} | Components used: {n_components}")

    # Extract the predictor row corresponding to the forecast origin
    try:
        X_t = X.loc[[last_observation_date]]
    except KeyError:
        # If exact date is missing, use last available observation
        X_t = X.iloc[[-1]]
        if verbose:
            print(f"Date {last_observation_date} not in X, using {X.index[-1]} instead.")

    X_t_scaled = scaler.transform(X_t)

    # Generate forecasts for each horizon
    pcr_forecasts = {}
    for h in range(forecast_horizon):
        forecast = pcr_models[h].predict(X_t_scaled)
        pcr_forecasts[h] = forecast[0]

    # Construct forecast dates starting one month after last observation
    start_date = pd.to_datetime(last_observation_date) + pd.DateOffset(months=1)
    forecast_dates = [start_date + pd.DateOffset(months=h) for h in pcr_forecasts.keys()]

    if verbose:
        print("\nForecasted months:")
        for date in forecast_dates:
            print(date.strftime("%Y-%m"))

    # Assemble forecast results into a DataFrame
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Inflation forecast": list(pcr_forecasts.values()),
        "Horizon": list(pcr_forecasts.keys())
    })

    return forecast_df



def run_rolling_forecast(X, y, forecast_horizon=12, start_date="2012-01", end_date="2015-12", window_length=72):
    """
    Runs a rolling real-time forecast using PCR (e.g., Principal Component Regression).
    One forecast is made each month, with predictions for the next `forecast_horizon` periods.

    Parameters:
    - X: DataFrame of explanatory variables.
    - y: Series of the target variable.
    - forecast_horizon: Number of steps ahead to forecast.
    - start_date: Start of the forecasting period.
    - end_date: End of the forecasting period.
    - window_length: Number of observations used in each rolling window.

    Returns:
    - DataFrame with all forecasts and metadata.
    """

    all_forecasts = []

    # Create list of monthly forecast dates
    forecast_dates = pd.date_range(start=start_date, end=end_date, freq="MS")

    for date in forecast_dates:
        # Select real-time training data up to the forecast date
        X_train = X.loc[:date]
        y_train = y.loc[:date]

        # Skip if not enough data for current window
        if len(X_train) < window_length:
            print("Skipping due to insufficient data.")
            continue

        # Fit scaler on the rolling window
        scaler = StandardScaler()
        scaler.fit(X_train.iloc[-window_length:])

        # Generate forecasts using a PCR-based model
        forecast_df = pcr_forecaster(
            X=X,
            y=y,
            forecast_horizon=forecast_horizon,
            last_observation_date=date,
            scaler=scaler,
            window_length=window_length, 
            variance_threshold=0.99,
            verbose=False
        )

        # Store the date the forecast was made
        forecast_df["Forecast_made_in"] = date
        all_forecasts.append(forecast_df)

    # Concatenate all forecasts into a single DataFrame
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