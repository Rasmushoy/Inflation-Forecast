import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def lasso_forecast_(X, y, last_observation_date, forecast_horizon=12):
    """
    Estimerer en LASSO-model for hvert forecast-horizon og forudsiger op til H måneder frem.
    
    Parametre:
        X_train: pd.DataFrame – allerede skaleret og trimmet feature-matrix
        y_train: pd.Series – allerede trimmet target
        last_observation_date: str – fx '2023-12-01'
        forecast_horizon: int – hvor mange måneder frem der forecastes

    Returnerer:
        forecast_df: pd.DataFrame med forecast
        lasso_models: dict med modeller
    """

    # 1. Skalering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    
    # 2. Træn modeller for 1 til forecast_horizon måneder frem
    lasso_models = {}
    forecasts = []
    start_date = pd.to_datetime(last_observation_date) + pd.DateOffset(months=1)

    for h in range(1, forecast_horizon):
        
        print(h)
        forecast_date = y.index.max()
        target_date = forecast_date + pd.DateOffset(months=h)
        print(f"Forudsigelsen for: {target_date.strftime('%Y-%m')}")
        
        
        y_shifted = y.shift(-h).dropna()
        X_h = X_scaled[:len(y_shifted), :]
        y_h = y_shifted

        model = LassoCV(cv=5, 
                        random_state=42, 
                        max_iter=100000)
        
        model.fit(X_h, y_h)
        
        lasso_models[h] = model

        coef = model.coef_
        alpha = model.alpha_
        num_zero = np.sum(coef == 0)
        num_nonzero = np.sum(coef != 0)

        print(f"Horizon {h:2d}: alpha = {alpha:.5f}, 0-koeff = {num_zero}, ≠0-koeff = {num_nonzero}")

        # Forecast med seneste datapunkt
        latest_data_scaled = scaler.transform(X_scaled.iloc[[-1]])
        forecast_value = model.predict(latest_data_scaled)[0]

        # Træningsfejl
        y_pred_train = model.predict(X_h)
        rmse = np.sqrt(mean_squared_error(y_h, y_pred_train))

        forecasts.append({
            "Dato": start_date + pd.DateOffset(months=h - 1),
            "Horizon": h,
            "Forecast": np.round(forecast_value, 3),
            "Train_RMSE": rmses
        })

    forecast_df = pd.DataFrame(forecasts)
    
    return forecast_df, lasso_models
