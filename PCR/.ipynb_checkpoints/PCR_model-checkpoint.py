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


def PCR_Forecaster(X, y, forecast_horizon, last_observation_date, scaler, variance_threshold=0.95, verbose=True):
    """
    Forecast inflation using Principal Component Regression (PCR) per forecast horizon.
    
    Args:
        X: DataFrame of predictors
        y: Series of target
        forecast_horizon: int, number of months ahead
        last_observation_date: str or Timestamp, point to forecast from
        scaler: fitted StandardScaler on training data
        variance_threshold: float, % of explained variance to retain (default: 0.95)
        verbose: print training info
    """

    # Trim data til real-time
    X = X.loc[:last_observation_date]
    y = y.loc[:last_observation_date]

    X_scaled = scaler.transform(X)
    
    pcr_models = {}

    for h in range(forecast_horizon):
        if verbose:
            print(f"\n=== Horisont h={h} ===")

        y_shifted = y.shift(-h).dropna()
        X_train = X_scaled[:len(y_shifted)]
        y_train = y_shifted

        # Automatisk valg af antal komponenter
        pca = PCA().fit(X_train)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumvar >= variance_threshold) + 1

        if verbose:
            print(f"Forklaret varians (k={n_components}): {cumvar[n_components-1]:.2%}")
            
        # PCR pipeline
        model = Pipeline([
            ("pca", PCA(n_components=n_components)),
            ("reg", LinearRegression())
        ])

        model.fit(X_train, y_train)
        pcr_models[h] = model

        if verbose:
            print(f"Træningsobs: {len(y_train)} | Komponenter: {n_components}")

    # Forudsig fra X_t
    try:
        X_t = X.loc[[last_observation_date]]
    except KeyError:
        X_t = X.iloc[[-1]]
        if verbose:
            print(f"Dato {last_observation_date} ikke i X, bruger {X.index[-1]} i stedet.")

    X_t_scaled = scaler.transform(X_t)

    # Forudsig
    pcr_forecasts = {}
    for h in range(forecast_horizon):
        forecast = pcr_models[h].predict(X_t_scaled)
        pcr_forecasts[h] = forecast[0]

    # Lav datoer
    start_date = pd.to_datetime(last_observation_date) + pd.DateOffset(months=1)
    forecast_dates = [start_date + pd.DateOffset(months=h) for h in pcr_forecasts.keys()]

    #  Print datoer for de forudsagte måneder
    print("\nForudsagte måneder:")
    for date in forecast_dates:
        print(date.strftime("%Y-%m"))
        
    forecast_df = pd.DataFrame({
        "Dato": forecast_dates,
        "Inflationsforecast": list(pcr_forecasts.values()),
        "Horizon": list(pcr_forecasts.keys())
    })

    return forecast_df


def pcr_forecast_rolling(X, y, forecast_horizon, last_observation_date, scaler, variance_threshold=0.95,window_length=108, verbose=True):
    """
    Forecast inflation using Principal Component Regression (PCR) per forecast horizon.
    
    Args:
        X: DataFrame of predictors
        y: Series of target
        forecast_horizon: int, number of months ahead
        last_observation_date: str or Timestamp, point to forecast from
        scaler: fitted StandardScaler on training data
        variance_threshold: float, % of explained variance to retain (default: 0.95)
        verbose: print training info
    """

    # Trim data til real-time
    X_init = X.loc[:last_observation_date]
    y_init = y.loc[:last_observation_date]
    
    X = X_init.iloc[-window_length:]
    y = y_init.iloc[-window_length:]

    X_scaled = scaler.transform(X)
    
    pcr_models = {}

    for h in range(forecast_horizon):
        if verbose:
            print(f"\n=== Horisont h={h} ===")

        y_shifted = y.shift(-h).dropna()
        X_train = X_scaled[:len(y_shifted)]
        y_train = y_shifted

        # Automatisk valg af antal komponenter
        pca = PCA().fit(X_train)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumvar >= variance_threshold) + 1

        if verbose:
            print(f"Forklaret varians (k={n_components}): {cumvar[n_components-1]:.2%}")
            
        # PCR pipeline
        model = Pipeline([
            ("pca", PCA(n_components=n_components)),
            ("reg", LinearRegression())
        ])

        model.fit(X_train, y_train)
        pcr_models[h] = model

        if verbose:
            print(f"Træningsobs: {len(y_train)} | Komponenter: {n_components}")

    # Forudsig fra X_t
    try:
        X_t = X.loc[[last_observation_date]]
    except KeyError:
        X_t = X.iloc[[-1]]
        if verbose:
            print(f"Dato {last_observation_date} ikke i X, bruger {X.index[-1]} i stedet.")

    X_t_scaled = scaler.transform(X_t)

    # Forudsig
    pcr_forecasts = {}
    for h in range(forecast_horizon):
        forecast = pcr_models[h].predict(X_t_scaled)
        pcr_forecasts[h] = forecast[0]

    # Lav datoer
    start_date = pd.to_datetime(last_observation_date) + pd.DateOffset(months=1)
    forecast_dates = [start_date + pd.DateOffset(months=h) for h in pcr_forecasts.keys()]

    if verbose:
        #  Print datoer for de forudsagte måneder
        print("\nForudsagte måneder:")
        for date in forecast_dates:
            print(date.strftime("%Y-%m"))
        
    forecast_df = pd.DataFrame({
        "Dato": forecast_dates,
        "Inflationsforecast": list(pcr_forecasts.values()),
        "Horizon": list(pcr_forecasts.keys())
    })

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
        forecast_df = pcr_forecast_rolling(
            X=X,
            y=y,
            forecast_horizon=forecast_horizon,
            last_observation_date=date,
            scaler=scaler,
            window_length=window_length, 
            variance_threshold=0.99,
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
