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



def lasso_forecast_rolling(X, y, forecast_horizon, last_observation_date, scaler, window_length=108, verbose=True):
    """
    Forecast inflation using one Random Forest model per horizon (direct forecast), based on Garcia et al. (2017)
    
    Args:
        X: DataFrame of predictors
        y: Series of target variable
        forecast_horizon: int, how many steps ahead to forecast (e.g. 12)
        last_observation_date: str or Timestamp, end of training data and point of forecast
        scaler: a fitted sklearn scaler (no data leakage!)
        window_length: int, number of time steps in rolling window (default 108 = 9 years of monthly data)
    """

    # 1. Begræns datasættet til kun at inkludere real-time tilgængelige observationer (op til last_observation_date)
    X = X.loc[:last_observation_date]
    y = y.loc[:last_observation_date]

    # 2. Definér rolling window
    if len(X) < window_length:
        raise ValueError("Not enough data for the chosen rolling window length.")
    
    X_window = X.iloc[-window_length:]
    y_window = y.iloc[-window_length:]

    # 3. Træn en model per horisont (direct forecast approach)
    lasso_models = {}

    for h in range(forecast_horizon):
        if verbose: 
            print(f"\n=== Horisont h={h+1} ===")
        
        # Laver y_{t+h}
        y_shifted = y_window.shift(-h).dropna()

        # Matcher X til y
        X_train = X_window.iloc[:len(y_shifted)]
        y_train = y_shifted
        
        tscv = TimeSeriesSplit(n_splits=4)
        alphas = np.logspace(-4, 1, 100)  # Fra 0.0001 til 10
        model = LassoCV(alphas=alphas, cv=tscv, max_iter=100000)

        model.fit(scaler.transform(X_train), y_train)

        lasso_models[h] = model

        #coef = model.coef_
      
        if verbose: 
            print(f"Antal træningsobservationer: {len(y_train)}")
            print(f"Valgt alpha: {model.alpha_:.5f}")
            print(f"0-koeff = {np.sum(model.coef_ == 0)}, ≠0-koeff = {np.sum(model.coef_ != 0)}")
            

    # 4. Lav forecast fra X_t (real-time available data)
    X_t = X.loc[[last_observation_date]]  # Real-time input
    X_t_scaled = scaler.transform(X_t)

    lasso_forecasts = {}

    for h in range(forecast_horizon):
        forecast = lasso_models[h].predict(X_t_scaled)
        lasso_forecasts[h] = forecast[0]  # Gem som float

    # 5. Generér datoer
    start_date = pd.to_datetime(last_observation_date) + pd.DateOffset(months=1)
    forecast_dates = [start_date + pd.DateOffset(months=h) for h in lasso_forecasts.keys()]

    forecast_df = pd.DataFrame({
        "Dato": forecast_dates,
        "Inflationsforecast": list(lasso_forecasts.values()),
        "Horizon": list(lasso_forecasts.keys())
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
        print(date)

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
        forecast_df = lasso_forecast_rolling(
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


############################################################################################################
############################################################################################################

def lasso_forecast(X, y, forecast_horizon, scaler, last_observation_date):
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