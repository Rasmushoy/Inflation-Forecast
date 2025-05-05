import os
import getpass
import numpy as np
import pandas as pd
from time import time
from functools import reduce
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



def lasso_forecast(X, y, forecast_horizon, last_observation_date, scaler, window_length=108, verbose=True, return_models=False):
    """
    Forecast inflation using one LASSO regression model per forecast horizon (direct forecast),
    based on the approach in Garcia et al. (2017).

    Args:
        X: DataFrame of predictor variables
        y: Series of the target variable (e.g., inflation)
        forecast_horizon: int, number of steps ahead to forecast (e.g., 12 months)
        last_observation_date: str or pd.Timestamp, point in time to generate forecast from
        scaler: a pre-fitted sklearn scaler (e.g., StandardScaler or MinMaxScaler)
        window_length: int, number of time steps in rolling training window (default = 108 months)
        verbose: bool, whether to print training progress and details
    """

    # --- STEP 1: Limit dataset to real-time available observations only ---
    X = X.loc[:last_observation_date]
    y = y.loc[:last_observation_date]

    # --- STEP 2: Define rolling window to simulate real-time training ---
    if len(X) < window_length:
        raise ValueError("Not enough data for the chosen rolling window length.")
    
    X_window = X.iloc[-window_length:]
    y_window = y.iloc[-window_length:]

    # --- STEP 3: Train one LASSO model per forecast horizon (direct forecast approach) ---
    lasso_models = {}

    for h in range(forecast_horizon):
        if verbose: 
            print(f"\n=== Horizon h={h+1} ===")

        # Shift target to create y_{t+h}
        y_shifted = y_window.shift(-h).dropna()

        # Align predictors with target
        X_train = X_window.iloc[:len(y_shifted)]
        y_train = y_shifted

        # Use time series cross-validation for hyperparameter tuning
        tscv = TimeSeriesSplit(n_splits=4)
        alphas = np.logspace(-4, 1, 100)  # Test alpha values from 0.0001 to 10

        # LASSO with cross-validation
        model = LassoCV(alphas=alphas, cv=tscv, max_iter=100000)
        model.fit(scaler.transform(X_train), y_train)

        # Store the trained model
        lasso_models[h] = model

        if verbose: 
            print(f"Number of training observations: {len(y_train)}")
            print(f"Selected alpha: {model.alpha_:.5f}")
            print(f"Zero coefficients: {np.sum(model.coef_ == 0)}, Non-zero coefficients: {np.sum(model.coef_ != 0)}")

    # --- STEP 4: Generate forecasts from most recent observation ---
    X_t = X.loc[[last_observation_date]]  # Use latest real-time data
    X_t_scaled = scaler.transform(X_t)

    lasso_forecasts = {}
    for h in range(forecast_horizon):
        forecast = lasso_models[h].predict(X_t_scaled)
        lasso_forecasts[h] = forecast[0]  # Save scalar prediction

    # --- STEP 5: Format forecast output with future dates ---
    start_date = pd.to_datetime(last_observation_date) + pd.DateOffset(months=1)
    forecast_dates = [start_date + pd.DateOffset(months=h) for h in lasso_forecasts.keys()]

    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Horizon": list(lasso_forecasts.keys()), 
        "Inflation forecast": list(lasso_forecasts.values())
    })

    if verbose: 
        print("\nForecasted months:")
        for date in forecast_dates:
            print(date.strftime("%Y-%m"))

    if return_models:
        return forecast_df, lasso_models
    else:
        return forecast_df


def run_rolling_forecast(X, y, forecast_horizon=12, start_date="2012-01", end_date="2015-12", window_length=72):
    """
    Performs rolling real-time forecasts using a forecasting function (e.g. LASSO),
    making one forecast per month, each with multiple horizons.

    Args:
        X (pd.DataFrame): Predictor variables.
        y (pd.Series): Target variable.
        forecast_horizon (int): How many steps ahead to forecast.
        start_date (str): Start of the forecast period (YYYY-MM).
        end_date (str): End of the forecast period (YYYY-MM).
        window_length (int): Number of months in the rolling training window.
        verbose (bool): If True, print forecast progress.

    Returns:
        pd.DataFrame: Combined forecast results with metadata.
    """

    all_forecasts = []
    forecast_dates = pd.date_range(start=start_date, end=end_date, freq="MS")

    for date in forecast_dates:

        # Real-time data available up to 'date'
        X_train = X.loc[:date]
        y_train = y.loc[:date]

        if len(X_train) < window_length:
            if verbose:
                print("Skipping due to insufficient training data.")
            continue

        # Scale training data (only the most recent window)
        scaler = StandardScaler()
        scaler.fit(X_train.iloc[-window_length:])

        # Forecast for specified horizon
        forecast_df = lasso_forecast(
            X=X,
            y=y,
            forecast_horizon=forecast_horizon,
            last_observation_date=date,
            scaler=scaler,
            window_length=window_length,
            verbose=False
        )

        forecast_df["Forecast_made_in"] = date  # When the forecast was made
        all_forecasts.append(forecast_df)

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


############################################################################################################
############################################################################################################

def lasso_forecast_OLD(X, y, forecast_horizon, scaler, last_observation_date):
    """
    Træner en LASSO-model til forecasting og forudsiger inflation op til forecast_horizon måneder frem.
    
    Parametre:
        X: pd.DataFrame – features
        y: pd.Series – target (inflation)
        last_observation_date: str – f.eks. '2024-12-01'
        forecast_horizon: int – antal måneder frem
        plot: bool – om der skal vises plot
    
    Returnerer:
        forecast_df: pd.DataFrame med datoer og forudsagte værdier
    """

    # 1. Split data
    X = X.loc[:last_observation_date]
    y = y.loc[:last_observation_date]
    
    X_scaled = scaler.transform(X)
    
    # 2. Træn modeller for 1 til forecast_horizon måneder frem
    lasso_models = {}

    #for h in range(1, forecast_horizon + 1):
    for h in range(0, forecast_horizon):
        print(f"\n=== Horizon {h} ===")

        y_shifted = y.shift(-h)
        y_shifted = y_shifted.dropna()
     
        #X_train = X[:len(y_shifted), :]
        X_train = X_scaled[:len(y_shifted)]

        y_train = y_shifted
        
        #print(y_shifted)
        
        tscv = TimeSeriesSplit(n_splits=4)
        alphas = np.logspace(-4, 1, 100)  # Fra 0.0001 til 10
        model = LassoCV(alphas=alphas, cv=tscv, max_iter=100000)

    
        #model = LassoCV(cv=tscv, random_state=42, max_iter=100000)
        #model = LassoCV(cv=5, random_state=42, max_iter=100000)
        #model.fit(X_train, y_train)
        model.fit(X_train, y_train)
        lasso_models[h] = model
        
        coef = model.coef_
        print(f"Antal træningsobservationer: {len(y_train)}")
        print(f"Valgt alpha: {model.alpha_:.5f}")
        print(f"0-koeff = {np.sum(coef == 0)}, ≠0-koeff = {np.sum(coef != 0)}")
    
    # 3. Forudsig fremtiden
    X_t = X.loc[[last_observation_date]]
    X_t_scaled = scaler.transform(X_t)

    
    forecasts = {}
    for h in range(0, forecast_horizon):
        forecast_value = lasso_models[h].predict(X_t_scaled)
        forecasts[h] = forecast_value[0]

    # 4. Datoetiketter
    start_date = pd.to_datetime(last_observation_date) + pd.DateOffset(months=1)
    forecast_dates = [start_date + pd.DateOffset(months=h) for h in forecasts.keys()]

    # 👇 Print datoer for de forudsagte måneder
    print("\nForudsagte måneder:")
    for date in forecast_dates:
        print(date.strftime("%Y-%m"))
        
        
    forecast_df = pd.DataFrame({
        "Dato": forecast_dates,
        "Inflationsforecast": list(forecasts.values()),
        "Horizon": list(forecasts.keys())
    })
   
    return forecast_df