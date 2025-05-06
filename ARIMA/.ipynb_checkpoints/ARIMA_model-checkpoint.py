import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(y, forecast_horizon, last_observation_date, order=(1, 1, 0), window_length=108, verbose=True):
    """
    Forecast inflation using ARIMA with a direct forecast approach (one model per horizon),
    using a rolling window of past data ending at last_observation_date.

    Args:
        y (pd.Series): Target variable (e.g., inflation), indexed by date.
        forecast_horizon (int): Number of steps ahead to forecast (e.g., 12 months).
        last_observation_date (str or Timestamp): The date up to which data is used for training.
        order (tuple): ARIMA(p,d,q) order.
        window_length (int): Number of time steps in training window.
        verbose (bool): Whether to print progress.

    Returns:
        forecast_df (pd.DataFrame): Forecasts for each horizon with corresponding future dates.
    """

    # --- STEP 1: Begræns data til rolling window ---
    y = y.loc[:last_observation_date]
    if len(y) < window_length:
        raise ValueError("Not enough data for the specified window length.")

    y_window = y.iloc[-window_length:]

    forecasts = {}

    for h in range(1, forecast_horizon + 1):  # Start fra 1-måned frem
        if verbose:
            print(f"\n=== Horizon h={h} ===")

        # Justér target så det forskydes h skridt frem
        y_shifted = y_window.shift(-h).dropna()

        # Træk observationslængde i overensstemmelse med shifted target
        y_train = y_shifted

        try:
            model = ARIMA(y_train, order=order)
            fitted_model = model.fit()

            forecast = fitted_model.forecast(steps=1)
            forecasts[h] = forecast.iloc[0]

            if verbose:
                print(f"Forecast: {forecast.iloc[0]:.3f}")
        except Exception as e:
            print(f"Failed to fit ARIMA at horizon {h}: {e}")
            forecasts[h] = np.nan

    # --- STEP 2: Tilføj fremtidige datoer og returnér som DataFrame ---
    start_date = pd.to_datetime(last_observation_date) + pd.DateOffset(months=1)
    forecast_dates = [start_date + pd.DateOffset(months=h-1) for h in forecasts.keys()]

    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Horizon": list(forecasts.keys()),
        "Inflation forecast": list(forecasts.values())
    })

    return forecast_df
