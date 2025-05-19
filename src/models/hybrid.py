# -*- coding: utf-8 -*-
"""
Implementation of Hybrid ARIMA+LSTM models.
"""

import numpy as np
import pandas as pd
import copy
import warnings

# Import TF availability flag and model types from lstm/arima modules
try:
    from .lstm import TF_AVAILABLE
    from .arima import ARIMAResultsWrapper, auto_arima, pm, calculate_arima_residuals, arima_trajectory_forecast
except ImportError:
    # Fallback for direct execution or different environment setup
    try:
        from lstm import TF_AVAILABLE
        from arima import ARIMAResultsWrapper, auto_arima, pm, calculate_arima_residuals, arima_trajectory_forecast
    except ImportError as e:
        print(f"Error importing sibling modules in hybrid.py: {e}")
        TF_AVAILABLE = False # Assume TF unavailable if imports fail
        ARIMAResultsWrapper = None
        auto_arima = None
        pm = None
        # Define dummy functions to avoid NameErrors
        def calculate_arima_residuals(*args, **kwargs): return None
        def arima_trajectory_forecast(*args, **kwargs): return None


# Import config for default parameters
try:
    from .. import config # Relative import from parent directory (src)
except ImportError:
    import config # Fallback for direct execution


def hybrid_rolling_forecast(train_val_series, test_series, arima_model, lstm_residual_model, lstm_residual_scaler, window_size=config.LSTM_WINDOW_SIZE):
    """
    Performs rolling forecast using a Hybrid ARIMA+LSTM model.

    Args:
        train_val_series (pd.Series): Training and validation data.
        test_series (pd.Series): The test time series data.
        arima_model (ARIMAResultsWrapper or pmdarima.arima.ARIMA): The pre-trained base ARIMA model.
        lstm_residual_model (tf.keras.models.Sequential): The pre-trained LSTM model for residuals.
        lstm_residual_scaler (MinMaxScaler): The scaler fitted on the ARIMA residuals.
        window_size (int): LSTM input sequence length. Defaults to config.LSTM_WINDOW_SIZE.

    Returns:
        pd.Series or None: The hybrid rolling forecast predictions, or None if forecasting fails.
    """
    if not TF_AVAILABLE or arima_model is None or lstm_residual_model is None:
        print("[Hybrid Model] Error: Missing prerequisites (TF, ARIMA model, or LSTM model) for rolling forecast.")
        return None

    print(f"\n[Hybrid Model] Starting Hybrid Rolling Forecast for {len(test_series)} steps...")
    try:
        # Calculate initial residuals on the training data using the base arima model
        # Note: calculate_arima_residuals now handles different model types
        initial_residuals = calculate_arima_residuals(train_val_series, arima_model)
        if initial_residuals is None:
            print("Failed to calculate initial residuals for hybrid forecast.")
            return None

        # Ensure enough residuals for the first window
        if len(initial_residuals) < window_size:
            print(f"Error: Not enough initial residuals ({len(initial_residuals)}) for LSTM window ({window_size}).")
            return None

        # Use actual residuals from train_val phase for initial LSTM history
        # Ensure correct shape for scaler
        if lstm_residual_scaler is not None:
            scaled_residuals_history = list(lstm_residual_scaler.transform(initial_residuals.values.reshape(-1, 1)).flatten())
        else:
            scaled_residuals_history = list(initial_residuals.values.flatten())
        price_history = list(train_val_series) # For fallback only if ARIMA prediction fails
        final_predictions = []
        test_index = test_series.index

        # Determine ARIMA model type for correct rolling update/extend
        is_statsmodels = ARIMAResultsWrapper is not None and isinstance(arima_model, ARIMAResultsWrapper)
        is_pmdarima = pm is not None and isinstance(arima_model, pm.arima.ARIMA)

        if not is_statsmodels and not is_pmdarima:
            print("Error: Unknown ARIMA model type provided to hybrid_rolling_forecast.")
            return None

        # Use copies/reassignment for models that modify state in-place or return new objects
        current_arima_model = arima_model
        if is_pmdarima:
            current_arima_model = copy.deepcopy(arima_model) # pmdarima update modifies in-place
        # For statsmodels, extend returns a new object, so we reassign it in the loop (current_arima_model = ...)

        for t in range(len(test_series)):
            arima_pred = np.nan
            # Get the actual observation for this step (as a Series with index)
            current_actual_observation = test_series.iloc[t:t+1] # Keep index!
            current_actual = current_actual_observation.iloc[0] # Get scalar value

            # 1. Predict next step with ARIMA (using potentially updated/extended model)
            try:
                if is_statsmodels:
                    arima_pred = current_arima_model.forecast(steps=1).iloc[0]
                elif is_pmdarima:
                    arima_pred = current_arima_model.predict(n_periods=1)[0]
                else:
                    raise ValueError("Unknown ARIMA model type during prediction")
            except Exception as e:
                 print(f"Warning: ARIMA prediction failed at step {t}. Type: {type(e).__name__}. Using fallback (last known price).")
                 arima_pred = price_history[-1] if price_history else 0.0 # Use scalar history for fallback

            # 2. Predict next step residual with LSTM (based on *actual* past residuals history)
            lstm_pred_resid = 0.0 # Default residual prediction
            if len(scaled_residuals_history) >= window_size:
                 current_resid_window = scaled_residuals_history[-window_size:]
                 # Reshape for LSTM input [samples, time steps, features]
                 input_seq_resid = np.array(current_resid_window).reshape((1, window_size, 1))
                 try:
                     lstm_pred_scaled = lstm_residual_model.predict(input_seq_resid, verbose=0)[0][0]
                     # Inverse transform the scaled residual prediction
                     if lstm_residual_scaler is not None:
                         lstm_pred_resid = lstm_residual_scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
                     else:
                         lstm_pred_resid = lstm_pred_scaled # Already in original (residual) scale
                 except Exception as lstm_e:
                     print(f"Warning: LSTM residual prediction failed at step {t}. Type: {type(lstm_e).__name__}. Using 0.")
                     lstm_pred_resid = 0.0
            else:
                 # This case should ideally not happen if initial check passed, but included for safety
                 print(f"Warning: Not enough residual history at step {t}. Using 0 for residual prediction.")

            # 3. Combine predictions
            final_pred = arima_pred + lstm_pred_resid
            final_predictions.append(final_pred)

            # 4. Update histories and ARIMA model state
            price_history.append(current_actual) # Append actual price to external list

            # Calculate actual residual for this step using the ARIMA prediction *for this step*
            actual_residual = current_actual - arima_pred
            try:
                # Scale the actual residual and append to history for next LSTM input
                if lstm_residual_scaler is not None:
                    scaled_actual_residual = lstm_residual_scaler.transform([[actual_residual]])[0][0]
                else:
                    scaled_actual_residual = actual_residual # Already in original (residual) scale
                scaled_residuals_history.append(scaled_actual_residual)
            except Exception as scale_e:
                 print(f"Warning: Failed to process actual residual at step {t}. Type: {type(scale_e).__name__}. Appending last known or 0.")
                 # Append a fallback value to keep the history length consistent
                 if scaled_residuals_history: scaled_residuals_history.append(scaled_residuals_history[-1])
                 else: scaled_residuals_history.append(0.0)

            # Update ARIMA model state *after* using it for prediction and calculating residual
            try:
                if is_statsmodels:
                    # Extend returns a new ResultsWrapper object.
                    current_arima_model = current_arima_model.extend(current_actual_observation)
                elif is_pmdarima:
                    # Update the pmdarima model state in-place
                    current_arima_model.update(current_actual_observation)
            except Exception as update_e:
                 print(f"Warning: ARIMA model update failed at step {t}. Type: {type(update_e).__name__}. Forecast quality may degrade.")
                 # Model state might be inconsistent for the next step

            if (t + 1) % 50 == 0 or t == len(test_series) - 1:
                 print(f"Hybrid Rolling forecast step {t+1}/{len(test_series)} completed.")

        print("[Hybrid Model] Hybrid Rolling Forecast finished.")
        # Ensure the length matches test_series
        if len(final_predictions) != len(test_series):
             print(f"Warning: Hybrid prediction length ({len(final_predictions)}) mismatch with test series ({len(test_series)}). Padding with NaNs.")
             final_predictions = [np.nan] * (len(test_series) - len(final_predictions)) + final_predictions
        return pd.Series(final_predictions, index=test_index)

    except Exception as e:
        print(f"An error occurred during Hybrid rolling forecast setup: {e}")
        import traceback
        traceback.print_exc()
        return None


def hybrid_trajectory_forecast(train_val_series, test_series, arima_model, lstm_residual_model, lstm_residual_scaler, window_size=config.LSTM_WINDOW_SIZE):
    """
    Performs trajectory forecast using a Hybrid ARIMA+LSTM model.
    Uses standard multi-step forecast for ARIMA part and iterative prediction for LSTM residuals.

    Args:
        train_val_series (pd.Series): Training and validation data.
        test_series (pd.Series): The test time series data (used for length and index).
        arima_model (ARIMAResultsWrapper or pmdarima.arima.ARIMA): The pre-trained base ARIMA model.
        lstm_residual_model (tf.keras.models.Sequential): The pre-trained LSTM model for residuals.
        lstm_residual_scaler (MinMaxScaler): The scaler fitted on the ARIMA residuals.
        window_size (int): LSTM input sequence length. Defaults to config.LSTM_WINDOW_SIZE.

    Returns:
        pd.Series or None: The hybrid trajectory forecast predictions, or None if forecasting fails.
    """
    if not TF_AVAILABLE or arima_model is None or lstm_residual_model is None:
        print("[Hybrid Model] Error: Missing prerequisites (TF, ARIMA model, or LSTM model) for trajectory forecast.")
        return None

    test_len = len(test_series)
    print(f"\n[Hybrid Model] Starting Hybrid Trajectory Forecast for {test_len} steps...")

    try:
        # 1. Get the full ARIMA trajectory forecast first
        # Note: arima_trajectory_forecast is imported from .arima
        arima_preds = arima_trajectory_forecast(train_val_series, test_series, arima_model)
        if arima_preds is None:
            print("Failed to get ARIMA trajectory for hybrid forecast.")
            return None
        # Ensure arima_preds is a numpy array for easier handling later if needed
        arima_preds_values = arima_preds.values if isinstance(arima_preds, pd.Series) else np.array(arima_preds)


        # 2. Initialize LSTM residual prediction history
        # Note: calculate_arima_residuals is imported from .arima
        initial_residuals = calculate_arima_residuals(train_val_series, arima_model) # Actual residuals from training phase
        if initial_residuals is None:
            print("Failed to calculate initial residuals for hybrid trajectory forecast.")
            return None
        # Ensure correct shape for scaler
        if lstm_residual_scaler is not None:
            scaled_initial_residuals = lstm_residual_scaler.transform(initial_residuals.values.reshape(-1, 1)).flatten()
        else:
            scaled_initial_residuals = initial_residuals.values.flatten()

        if len(scaled_initial_residuals) < window_size:
             print(f"Error: Not enough initial residuals ({len(scaled_initial_residuals)}) to form LSTM window ({window_size}). Cannot start trajectory.")
             return None

        # History window for LSTM residual trajectory (starts with actual scaled residuals)
        scaled_residuals_history_window = list(scaled_initial_residuals[-window_size:])
        final_predictions = []

        for t in range(test_len):
            # 3. Predict next step residual with LSTM (using predicted residual history)
            lstm_pred_resid = 0.0
            lstm_pred_scaled = 0.0 # Initialize scaled prediction
            try:
                # Reshape window for LSTM input
                input_seq_resid = np.array(scaled_residuals_history_window).reshape((1, window_size, 1))
                lstm_pred_scaled = lstm_residual_model.predict(input_seq_resid, verbose=0)[0][0]
                # Inverse transform the scaled residual prediction
                if lstm_residual_scaler is not None:
                    lstm_pred_resid = lstm_residual_scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
                else:
                    lstm_pred_resid = lstm_pred_scaled # Already in original (residual) scale
            except Exception as lstm_e:
                 print(f"Warning: LSTM residual trajectory prediction failed at step {t}. Type: {type(lstm_e).__name__}. Using 0.")
                 lstm_pred_resid = 0.0
                 lstm_pred_scaled = 0.0 # Ensure scaled value is also 0 if prediction fails

            # 4. Combine ARIMA prediction (already computed for all steps) and LSTM residual prediction
            # Ensure arima_preds_values[t] is accessed correctly
            current_arima_pred = arima_preds_values[t] if t < len(arima_preds_values) else np.nan
            if np.isnan(current_arima_pred):
                 print(f"Warning: ARIMA prediction is NaN at step {t}. Final prediction will be NaN.")
                 final_pred = np.nan
            else:
                 final_pred = current_arima_pred + lstm_pred_resid
            final_predictions.append(final_pred)

            # 5. Update LSTM residual history window with the *predicted* scaled residual
            # (Maintaining original logic as per user decision)
            scaled_residuals_history_window.pop(0)
            scaled_residuals_history_window.append(lstm_pred_scaled) # Append predicted scaled residual

            if (t + 1) % 50 == 0 or t == test_len - 1:
                 print(f"Hybrid Trajectory forecast step {t+1}/{test_len} completed.")

        print("[Hybrid Model] Hybrid Trajectory Forecast finished.")
        # Return as Pandas Series with correct index
        return pd.Series(final_predictions, index=test_series.index)

    except Exception as e:
        print(f"An error occurred during Hybrid trajectory forecast: {e}")
        import traceback
        traceback.print_exc()
        return None