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

def XGBoost_Forecaster_Rolling(X, y, forecast_horizon, last_observation_date, scaler, window_length=108, verbose=True):
    """
    Forecast inflation using one XGBoost model per horizon (direct forecast), based on Garcia et al. (2017)
    
    Args:
        X: DataFrame of predictors
        y: Series of target variable
        forecast_horizon: int, how many months ahead to forecast
        last_observation_date: str or Timestamp, point of forecast
        scaler: fitted sklearn scaler
        window_length: int, rolling window length (default: 108 months)
        verbose: print training details
    """

    # 1. Begræns data til real-time
    X = X.loc[:last_observation_date]
    y = y.loc[:last_observation_date]

    if len(X) < window_length:
        raise ValueError("Not enough data for the rolling window.")

    # 2. Træk rolling window
    X_window = X.iloc[-window_length:]
    y_window = y.iloc[-window_length:]

    xgb_models = {}

    # 3. Træn én XGBoost-model per horisont
    for h in range(forecast_horizon):
        if verbose: 
            print(f"\n=== Horisont h={h+1} ===")
        
        y_shifted = y_window.shift(-h).dropna()
        X_train = X_window.iloc[:len(y_shifted)]
        y_train = y_shifted

        # Skalér input
        X_train_scaled = scaler.transform(X_train)

        # Brug evt. dine tuned parametre her
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
            print(f"Antal træningsobservationer: {len(y_train)}")
            print(f"Antal features: {X_train.shape[1]}")

    # 4. Lav forecast fra seneste datapunkt
    X_t = X.loc[[last_observation_date]]
    X_t_scaled = scaler.transform(X_t)

    xgb_forecasts = {}
    for h in range(forecast_horizon):
        forecast = xgb_models[h].predict(X_t_scaled)
        xgb_forecasts[h] = forecast[0]

    # 5. Formatér output
    start_date = pd.to_datetime(last_observation_date) + pd.DateOffset(months=1)
    forecast_dates = [start_date + pd.DateOffset(months=h) for h in xgb_forecasts.keys()]

    forecast_df = pd.DataFrame({
        "Dato": forecast_dates,
        "Inflationsforecast": list(xgb_forecasts.values()),
        "Horizon": list(xgb_forecasts.keys())
    })

    if verbose:
        print("\nForudsagte måneder:")
        for date in forecast_dates:
            print(date.strftime("%Y-%m"))

    return forecast_df


def run_rolling_forecast(X, y, forecast_horizon=12, start_date="2012-01", end_date="2015-12", window_length=72):
    """
    Kører rolling real-time forecast med Random Forest, én forecast per måned (med 12 horisonter per gang)
    """

    all_forecasts = []

    forecast_dates = pd.date_range(start=start_date, end=end_date, freq="MS")

    for date in forecast_dates:
        print(f"\n=== Forecast lavet i: {date.strftime('%Y-%m')} ===")

        # Real-time datasæt
        X_train = X.loc[:date]
        y_train = y.loc[:date]

        if len(X_train) < window_length:
            print("Springer over pga. for lidt data.")
            continue

        # Skaler træningsdata (rolling window)
        scaler = StandardScaler()
        scaler.fit(X_train.iloc[-window_length:])

        # Lav forecast for 12 horisonter
        forecast_df = XGBoost_Forecaster_Rolling(
            X=X,
            y=y,
            forecast_horizon=forecast_horizon,
            last_observation_date=date,
            scaler=scaler,
            window_length=window_length, 
            verbose=False
        )


        forecast_df["Forecast_made_in"] = date  # hvornår forecast blev lavet
        all_forecasts.append(forecast_df)

    all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    return all_forecasts_df


def evaluate_forecasts(forecast_df, y):
    """
    Matcher forecast med y_true og returnerer RMSE og MAE pr. horisont
    """
    # Merge forecast og faktisk inflation
    merged = forecast_df.merge(
        y.rename("y_true"), 
        left_on="Dato", 
        right_index=True,
        how="left"
    )

    # Filtrer ud fremtid hvor vi ikke har faktisk inflation
    merged = merged.dropna(subset=["y_true"])

    # Beregn fejl
    merged["error"] = merged["Inflationsforecast"] - merged["y_true"]
    merged["abs_error"] = merged["error"].abs()
    merged["squared_error"] = merged["error"] ** 2

    # Evaluer pr. horisont
    evaluation = merged.groupby("Horizon").agg(
        MAE=("abs_error", "mean"),
        RMSE=("squared_error", lambda x: (x.mean())**0.5),
        N_obs=("y_true", "count")
    ).reset_index()

    return evaluation, merged
