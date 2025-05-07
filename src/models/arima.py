# -*- coding: utf-8 -*-
"""
Implementation of ARIMA and Auto ARIMA models.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper
import copy
import warnings

# Suppress specific warnings if needed
warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')
warnings.filterwarnings("ignore", category=FutureWarning, module='statsmodels')


# Handle pmdarima import separately
try:
    import pmdarima as pm
    from pmdarima import auto_arima
except ImportError:
    print("Warning: pmdarima not found. Auto ARIMA functionality will be unavailable.")
    print("Install it using: pip install pmdarima")
    auto_arima = None # Set to None if not available
    pm = None # Ensure pm is None if import fails

# Import config for default parameters
try:
    from .. import config # Relative import from parent directory (src)
except ImportError:
    import config # Fallback for direct execution

def train_arima(series, order=(1, 1, 1)):
    """
    Trains a standard ARIMA model with a fixed order. Includes constant term.

    Args:
        series (pd.Series): The time series data to train on.
        order (tuple): The (p, d, q) order for the ARIMA model.

    Returns:
        ARIMAResultsWrapper or None: The fitted statsmodels ARIMA model, or None if training fails.
    """
    print(f"\n[ARIMA Model] Training ARIMA{order}...")
    try:
        # Ensure series has a DatetimeIndex or PeriodIndex with frequency
        if not isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
             print("Warning: ARIMA training requires DatetimeIndex or PeriodIndex. Attempting to infer frequency.")
             series = series.asfreq(pd.infer_freq(series.index)) # Try to infer frequency
        # statsmodels ARIMA requires a frequency. If it's missing after preprocessing, fail training.
        elif series.index.freq is None:
             print("Error: ARIMA training data index has no frequency after preprocessing. Aborting.")
             return None

        # Proceed only if frequency is set
        # Reverted: trend='c' is incompatible with d=1 in statsmodels ARIMA
        # If d=0, trend='c' is valid. If d>0, use trend='t' for linear trend equivalent.
        # For a generic (1,1,1) as requested, we don't add trend here.
        # If a drift is desired for d=1, it's implicitly handled by the differencing.
        model = ARIMA(series, order=order) # No trend='c' for d=1
        fitted_model = model.fit()
        print(f"Successfully trained ARIMA{order} model.")
        try:
            print(f"\n--- ARIMA{order} Model Summary ---")
            print(fitted_model.summary())
            print("-" * 30)
        except Exception as summary_e:
            print(f"Could not print ARIMA{order} summary: {summary_e}")
        return fitted_model
    except Exception as e:
        print(f"Error training ARIMA{order} model: {e}")
        return None

def train_auto_arima(series, seasonal=config.AUTO_ARIMA_SEASONAL, **kwargs):
    """
    Trains an ARIMA model by automatically selecting the best order using pmdarima.
    Includes intercept search.

    Args:
        series (pd.Series): The time series data to train on.
        seasonal (bool): Whether to consider seasonal components. Defaults to config.AUTO_ARIMA_SEASONAL.
        **kwargs: Additional arguments passed to pmdarima.auto_arima.

    Returns:
        pmdarima.arima.ARIMA or None: The fitted pmdarima model, or None if training fails or pmdarima not installed.
    """
    print("\n[ARIMA Model] Training Auto ARIMA...")
    if auto_arima is None:
        print("Error: pmdarima is not installed. Cannot perform Auto ARIMA.")
        return None
    try:
        # pmdarima is generally less strict about frequency but works better with it
        default_kwargs = {
            'start_p': config.AUTO_ARIMA_START_P, 'start_q': config.AUTO_ARIMA_START_Q,
            'max_p': config.AUTO_ARIMA_MAX_P, 'max_q': config.AUTO_ARIMA_MAX_Q,
            'd': None, # Let auto_arima determine d
            'seasonal': seasonal,
            'stepwise': True,
            'suppress_warnings': True,
            'error_action': 'ignore',
            'trace': config.AUTO_ARIMA_TRACE,
            'with_intercept': True # Add intercept (drift) term search
        }
        # Update default kwargs with any user-provided kwargs
        default_kwargs.update(kwargs)

        # Pass series directly, pmdarima handles numpy/pandas
        model = auto_arima(series, **default_kwargs)
        print(f"Successfully trained Auto ARIMA model. Best order: {model.order}, Seasonal order: {model.seasonal_order}")
        try:
            print("\n--- Auto ARIMA Model Summary ---")
            print(model.summary())
            print("-" * 30)
        except Exception as summary_e:
            print(f"Could not print Auto ARIMA summary: {summary_e}")
        return model
    except Exception as e:
        print(f"Error training Auto ARIMA model: {e}")
        return None

def arima_rolling_forecast(train_val_series, test_series, model):
    """
    Performs rolling forecast using a pre-trained ARIMA model (statsmodels or pmdarima).
    Uses extend (statsmodels) or update (pmdarima) for state updates.

    Args:
        train_val_series (pd.Series): Training and validation data (not directly used, but context).
        test_series (pd.Series): The test time series data.
        model (ARIMAResultsWrapper or pmdarima.arima.ARIMA): The pre-trained ARIMA model.

    Returns:
        pd.Series or None: The rolling forecast predictions, or None if forecasting fails.
    """
    predictions = []
    test_index = test_series.index

    print(f"\n[ARIMA Model] Starting ARIMA Rolling Forecast for {len(test_series)} steps...")

    is_statsmodels = isinstance(model, ARIMAResultsWrapper)
    is_pmdarima = pm is not None and isinstance(model, pm.arima.ARIMA)

    if not is_statsmodels and not is_pmdarima:
        print("Error: Unknown ARIMA model type provided to arima_rolling_forecast.")
        return None

    # Use a copy for pmdarima as update modifies in-place
    # For statsmodels, extend returns a new object, so we reassign in the loop
    current_model = model
    if is_pmdarima:
        current_model = copy.deepcopy(model)
    elif is_statsmodels:
        # statsmodels extend requires the original model object to be passed each time
        # We will use the .append method which modifies the model in-place if available,
        # otherwise fall back to extend (which is slower as it refits).
        # Note: .append might not be available in all statsmodels versions or for all model types.
        # Let's stick to extend for broader compatibility, accepting the performance hit.
        # The state is carried within the `current_model` object which gets replaced by extend's output.
        pass # No deepcopy needed, extend returns new object


    for t in range(len(test_series)):
        yhat = np.nan
        # Get the actual observation for this step (as a Series with index)
        current_actual_observation = test_series.iloc[t:t+1] # Keep index!

        # Predict step
        try:
            if is_statsmodels:
                # Forecast 1 step ahead from the current state
                yhat = current_model.forecast(steps=1).iloc[0]
            elif is_pmdarima:
                # Predict 1 period ahead
                yhat = current_model.predict(n_periods=1)[0]
            predictions.append(yhat)
        except Exception as pred_e:
            print(f"Error during prediction at step {t}: {type(pred_e).__name__} - {pred_e}. Appending NaN.")
            predictions.append(np.nan)
            # Skip update if prediction failed
            continue # Go to next iteration

        # Update step (only if prediction succeeded)
        try:
            if is_statsmodels:
                 # Extend the model with the new observation. extend() re-estimates parameters.
                 # This is computationally expensive for rolling forecasts.
                 # Consider using `append` if available and parameter constancy is assumed.
                 # For now, using extend as per original logic.
                 current_model = current_model.extend(current_actual_observation)
            elif is_pmdarima:
                 # Update the pmdarima model state using the Series
                 current_model.update(current_actual_observation)
        except Exception as update_e:
             # Handle update error separately from prediction error
             print(f"Error during model update at step {t}: {type(update_e).__name__} - {update_e}. Forecast quality may degrade.")
             # Model state might be inconsistent for the next step.
             pass # Continue loop, but model state is stale

        if (t + 1) % 50 == 0 or t == len(test_series) - 1:
            print(f"Rolling forecast step {t+1}/{len(test_series)} completed.")

    print("[ARIMA Model] ARIMA Rolling Forecast finished.")
    # Ensure the length matches test_series, handle potential NaNs at start/due to errors
    if len(predictions) != len(test_series):
         print(f"Warning: Prediction length ({len(predictions)}) mismatch with test series ({len(test_series)}). Padding with NaNs.")
         # Pad with NaNs at the beginning if prediction is shorter
         predictions = [np.nan] * (len(test_series) - len(predictions)) + predictions

    return pd.Series(predictions, index=test_index)


def arima_trajectory_forecast(train_val_series, test_series, model):
    """
    Performs a trajectory forecast using the model's standard multi-step forecast method.

    Args:
        train_val_series (pd.Series): Training and validation data (used for context/fitting).
        test_series (pd.Series): The test time series data (used for length and index).
        model (ARIMAResultsWrapper or pmdarima.arima.ARIMA): The pre-trained ARIMA model.

    Returns:
        pd.Series or None: The trajectory forecast predictions, or None if forecasting fails.
    """
    test_len = len(test_series)
    print(f"\n[ARIMA Model] Starting ARIMA Trajectory Forecast for {test_len} steps...")
    is_statsmodels = isinstance(model, ARIMAResultsWrapper)
    is_pmdarima = pm is not None and isinstance(model, pm.arima.ARIMA)

    if is_statsmodels:
        try:
            # Use get_forecast which handles multi-step prediction based on the fitted model
            forecast_result = model.get_forecast(steps=test_len)
            predictions = forecast_result.predicted_mean # This is a Pandas Series
            # Ensure the index matches the test set index
            predictions.index = test_series.index # Assign test index
            print("[ARIMA Model] Statsmodels Trajectory Forecast finished.")
            return predictions
        except Exception as e:
            print(f"Error during statsmodels trajectory forecast: {type(e).__name__} - {e}")
            return None
    elif is_pmdarima:
        try:
            predictions_array = model.predict(n_periods=test_len)
            # Convert numpy array to Pandas Series with the correct index
            predictions = pd.Series(predictions_array, index=test_series.index)
            print("[ARIMA Model] pmdarima Trajectory Forecast finished.")
            return predictions
        except Exception as e:
            print(f"Error during pmdarima trajectory forecast: {type(e).__name__} - {e}")
            return None
    else:
        print("Error: Unknown ARIMA model type provided.")
        return None

def calculate_arima_residuals(series, arima_model):
    """
    Calculates the residuals of an ARIMA model on the series it was trained on.

    Args:
        series (pd.Series): The time series data the model was trained on.
        arima_model (ARIMAResultsWrapper or pmdarima.arima.ARIMA): The fitted ARIMA model.

    Returns:
        pd.Series or None: The residuals, or None if calculation fails.
    """
    print("\n[ARIMA Model] Calculating ARIMA residuals...")
    try:
        if isinstance(arima_model, ARIMAResultsWrapper):
            # Ensure series index matches model's index if possible
            # Use residuals attribute directly if available and aligned
            if hasattr(arima_model, 'resid') and arima_model.resid.index.equals(series.index):
                 residuals = arima_model.resid
            else:
                # Fallback to predicting in-sample
                predictions = arima_model.predict(start=series.index[0], end=series.index[-1])
                predictions = predictions.reindex(series.index) # Align index just in case
                residuals = series - predictions
        elif pm is not None and isinstance(arima_model, pm.arima.ARIMA):
            # pmdarima predict_in_sample returns numpy array, needs index
            # Alternatively, use residuals() method if available
            if hasattr(arima_model, 'resid') and callable(arima_model.resid):
                 residuals_array = arima_model.resid()
                 residuals = pd.Series(residuals_array, index=series.index[-len(residuals_array):]) # Align index from end
                 residuals = residuals.reindex(series.index) # Ensure full alignment
            else:
                predictions = arima_model.predict_in_sample()
                residuals = series - pd.Series(predictions, index=series.index[-len(predictions):]) # Align index from end
                residuals = residuals.reindex(series.index) # Ensure full alignment

        else:
            print("Error: Unknown ARIMA model type for residual calculation.")
            return None
        print("Successfully calculated ARIMA residuals.")
        return residuals.dropna()
    except Exception as e:
        print(f"Error calculating ARIMA residuals. Type: {type(e).__name__}, Args: {e.args}")
        return None