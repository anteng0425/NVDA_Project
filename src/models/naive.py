# -*- coding: utf-8 -*-
"""
Implementation of Naive forecasting methods.
"""

import pandas as pd
import numpy as np

def naive_forecast_rolling(test_series):
    """
    Performs a rolling naive forecast (predicts tomorrow's price as today's actual price).

    Args:
        test_series (pd.Series): The test time series data.

    Returns:
        pd.Series: The rolling naive forecast predictions.
    """
    # Predict today's price as yesterday's actual price
    predictions = test_series.shift(1)
    print("[Naive Model] Generated Naive Rolling Forecast.")
    return predictions # First value will be NaN

def naive_forecast_trajectory(train_val_series, test_series):
    """
    Performs a trajectory naive forecast (predicts all future prices as the last known actual price).

    Args:
        train_val_series (pd.Series): The training and validation time series data (used to get the last value).
        test_series (pd.Series): The test time series data (used for index).

    Returns:
        pd.Series: The trajectory naive forecast predictions.
    """
    if train_val_series is None or train_val_series.empty:
        print("[Naive Model] Error: train_val_series is empty for trajectory forecast.")
        return pd.Series(index=test_series.index) # Return empty series with correct index

    last_value = train_val_series.iloc[-1]
    predictions_array = np.full(len(test_series), last_value)
    predictions = pd.Series(predictions_array, index=test_series.index) # Create Series with index
    print(f"[Naive Model] Generated Naive Trajectory Forecast (all values = {last_value:.2f}).")
    return predictions