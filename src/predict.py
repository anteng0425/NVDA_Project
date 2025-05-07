# -*- coding: utf-8 -*-
"""
NVDA Stock Price Prediction Analysis using ARIMA, LSTM, and Hybrid Models.

This script implements and compares various time series forecasting models
for predicting NVIDIA (NVDA) stock prices based on the requirements outlined
in the project documentation.
"""

# Standard Libraries
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from math import sqrt

# Data Handling & Preprocessing
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split # Not explicitly used for time series split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# ARIMA Models
# Make sure pmdarima and statsmodels are installed:
# pip install pmdarima statsmodels
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper
try:
    import pmdarima as pm
    from pmdarima import auto_arima
except ImportError:
    print("Warning: pmdarima not found. Auto ARIMA functionality will be unavailable.")
    print("Install it using: pip install pmdarima")
    auto_arima = None # Set to None if not available

# LSTM Models
# Make sure tensorflow is installed:
# pip install tensorflow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    # Explicitly set memory growth if using GPU to avoid potential issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
except ImportError:
    print("Warning: tensorflow not found. LSTM and Hybrid model functionality will be unavailable.")
    print("Install it using: pip install tensorflow")
    tf = None # Set to None if not available

# Configuration
warnings.filterwarnings("ignore") # Ignore harmless warnings
plt.style.use('fivethirtyeight') # Use a common plotting style

# --- Constants ---
# Path relative to the src directory
CSV_PATH = '../data/raw/NVDA_stock_data_new.csv'
CUTOFF_DATE = '2023-03-14'
TRAIN_VAL_RATIO = 0.8
TRAIN_RATIO = 0.8
LSTM_WINDOW_SIZE = 20
LSTM_EPOCHS = 300 # As per documentation, though might be long
LSTM_BATCH_SIZE = 128
LSTM_PATIENCE = 30
LSTM_UNITS_1 = 32
LSTM_UNITS_2 = 16
LSTM_DENSE_UNITS = 16
LSTM_DROPOUT_RATE = 0.2

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(csv_path, cutoff_date):
    """
    Loads NVDA stock data from a CSV file, preprocesses it according
    to the project requirements. Sets 'date' as index.

    Args:
        csv_path (str): Path to the CSV file.
        cutoff_date (str): The date (YYYY-MM-DD) before which data should be kept.

    Returns:
        pd.DataFrame: Preprocessed data with DatetimeIndex and 'adj_close' column,
                      sorted by date, or None if loading fails.
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from {csv_path}")

        # Drop the first row if it's a duplicate header (as per documentation)
        if not df.empty and 'Date' in df.columns and df.iloc[0]['Date'] == 'Date': # Check if first row looks like header
             df = df.drop(index=0).reset_index(drop=True)
             print("Dropped duplicate header row.")

        # Select and rename columns
        if 'Date' in df.columns and 'Adj Close' in df.columns:
            df = df[['Date', 'Adj Close']].copy()
            df.rename(columns={'Date': 'date', 'Adj Close': 'adj_close'}, inplace=True)
            print("Selected and renamed 'Date' and 'Adj Close' columns.")
        else:
            print("Error: 'Date' or 'Adj Close' column not found in the CSV.")
            return None

        # Convert data types and handle errors
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['adj_close'] = pd.to_numeric(df['adj_close'], errors='coerce')

        # Drop rows with conversion errors (NaNs)
        initial_rows = len(df)
        df.dropna(subset=['date', 'adj_close'], inplace=True)
        if len(df) < initial_rows:
            print(f"Dropped {initial_rows - len(df)} rows due to invalid date or price data.")

        # Filter by date
        df = df[df['date'] < pd.to_datetime(cutoff_date)]
        print(f"Filtered data to before {cutoff_date}.")

        # Sort by date and set 'date' column as index
        df = df.sort_values(by='date')
        df.set_index('date', inplace=True)
        print(f"Data sorted and 'date' column set as index. Final shape: {df.shape}")

        if df.empty:
            print("Warning: DataFrame is empty after preprocessing.")
            return None

        # Ensure the DataFrame has a Business Day frequency ('B') for statsmodels
        try:
            # Attempt to set frequency to Business Day
            df = df.asfreq('B')
            print("Set DataFrame frequency to 'B' (Business Day).")
            # Check if resampling introduced NaNs and forward fill them
            if df['adj_close'].isnull().any():
                nan_count = df['adj_close'].isnull().sum()
                print(f"Warning: {nan_count} NaNs introduced after setting frequency to 'B'. Forward filling...")
                df['adj_close'].fillna(method='ffill', inplace=True)
                # Drop any remaining NaNs at the very beginning if ffill couldn't fill them
                df.dropna(inplace=True)
                print(f"NaNs filled. Final shape after frequency setting: {df.shape}")
        except ValueError as freq_error:
             # This might happen if dates are not unique or other index issues
             print(f"Warning: Could not set frequency to 'B'. Error: {freq_error}. ARIMA models might fail.")
             # Proceed without frequency if setting fails

        return df

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading/preprocessing: {e}")
        return None

# --- Data Splitting ---
def split_data(df, train_val_ratio=0.8, train_ratio=0.8):
    """
    Splits the dataframe (with DatetimeIndex) into training, validation,
    and test sets based on ratios.

    Args:
        df (pd.DataFrame): The preprocessed dataframe with DatetimeIndex.
        train_val_ratio (float): The proportion of data for training + validation.
        train_ratio (float): The proportion of the train+val set for actual training.

    Returns:
        tuple: Contains train_df, val_df, test_df, train_val_df (all pd.DataFrames).
               Returns (None, None, None, None) if splitting fails or df is invalid.
    """
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        print("Error: Invalid DataFrame (must have DatetimeIndex) provided for splitting.")
        return None, None, None, None

    try:
        # Data should already be sorted by index
        # Split into train+validation and test sets
        tv_size = int(len(df) * train_val_ratio)
        if tv_size <= 0 or tv_size >= len(df):
             print(f"Error: Invalid train_val_ratio ({train_val_ratio}) resulting in non-positive or full-size split.")
             return None, None, None, None

        train_val_df = df.iloc[:tv_size]
        test_df = df.iloc[tv_size:]
        print(f"Split into Train+Validation ({len(train_val_df)} rows) and Test ({len(test_df)} rows).")
        print(f"Test set date range: {test_df.index.min()} to {test_df.index.max()}")

        # Split train+validation into actual train and validation sets
        t_size = int(len(train_val_df) * train_ratio)
        if t_size <= 0 or t_size >= len(train_val_df):
             print(f"Error: Invalid train_ratio ({train_ratio}) resulting in non-positive or full-size split for train/val.")
             return None, None, None, None

        train_df = train_val_df.iloc[:t_size]
        val_df = train_val_df.iloc[t_size:]
        print(f"Split Train+Validation into Train ({len(train_df)} rows) and Validation ({len(val_df)} rows).")
        print(f"Train set date range: {train_df.index.min()} to {train_df.index.max()}")
        print(f"Validation set date range: {val_df.index.min()} to {val_df.index.max()}")

        return train_df, val_df, test_df, train_val_df

    except Exception as e:
        print(f"An unexpected error occurred during data splitting: {e}")
        return None, None, None, None

# --- Evaluation Metrics ---
def calculate_rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error (RMSE)."""
    return sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0: return np.inf
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def calculate_direction_accuracy(y_true, y_pred):
    """
    Calculates the Directional Accuracy (ACC).
    Compares the sign of the change from the previous day.
    Handles both Series and numpy arrays for y_pred.
    """
    # Ensure inputs are numpy arrays for consistent diff calculation
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)

    # Handle potential NaNs introduced by shift or errors
    valid_mask = ~np.isnan(y_true_array) & ~np.isnan(y_pred_array)
    y_true_valid = y_true_array[valid_mask]
    y_pred_valid = y_pred_array[valid_mask]


    if len(y_pred_valid) != len(y_true_valid):
         # This case might occur if NaNs are not aligned
         print(f"Warning: Length mismatch after NaN handling in direction accuracy. y_true_valid: {len(y_true_valid)}, y_pred_valid: {len(y_pred_valid)}. Returning 0.")
         return 0.0

    if len(y_true_valid) < 2: # Need at least 2 points to calculate diff
        return 0.0

    # Calculate differences on valid data
    true_diff = np.diff(y_true_valid)
    # Predict difference based on prediction and *previous prediction* (as per Markdown)
    pred_diff = np.diff(y_pred_valid) # Compare change in prediction with change in actual

    # Ensure lengths match after diff calculation
    min_len = min(len(true_diff), len(pred_diff))
    if min_len <= 0: return 0.0 # Return 0 if no comparable differences

    # Compare the sign of the differences
    correct_direction = (np.sign(true_diff[:min_len]) == np.sign(pred_diff[:min_len]))
    return np.mean(correct_direction) * 100

def calculate_r2(y_true, y_pred):
    """Calculates the R-squared (Coefficient of Determination)."""
    return r2_score(y_true, y_pred)

def evaluate_performance(model_name, y_true, y_pred):
    """Calculates and returns all performance metrics after handling NaNs."""
    # Ensure y_true and y_pred are numpy arrays for consistent processing
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Create mask for valid (non-NaN) pairs
    valid_mask = ~np.isnan(y_true_arr) & ~np.isnan(y_pred_arr)

    if np.sum(valid_mask) < 2: # Need at least 2 points for most metrics
        print(f"Could not evaluate {model_name} due to insufficient valid data points ({np.sum(valid_mask)}).")
        return {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

    y_true_valid = y_true_arr[valid_mask]
    y_pred_valid = y_pred_arr[valid_mask]

    try:
        rmse = calculate_rmse(y_true_valid, y_pred_valid)
        mape = calculate_mape(y_true_valid, y_pred_valid)
        # Pass the original arrays to ACC, it handles NaNs internally now
        acc = calculate_direction_accuracy(y_true_arr, y_pred_arr)
        r2 = calculate_r2(y_true_valid, y_pred_valid)

        print(f"--- {model_name} Performance ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.4f}%")
        print(f"Direction Accuracy (ACC): {acc:.4f}%")
        print(f"R-squared (R2): {r2:.4f}")
        print("-" * (len(model_name) + 22))
        return {"RMSE": rmse, "MAPE": mape, "ACC": acc, "R2": r2}
    except Exception as e:
        print(f"Error during evaluation for {model_name}: {e}")
        return {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}


# --- Model Implementation: Naive Forecast ---
def naive_forecast_rolling(test_series):
    """
    Performs a rolling naive forecast (predicts tomorrow's price as today's actual price).
    """
    predictions = test_series.shift(1)
    print("Generated Naive Rolling Forecast.")
    return predictions # First value is NaN

def naive_forecast_trajectory(train_val_series, test_series): # Pass test_series for index
    """
    Performs a trajectory naive forecast (predicts all future prices as the last known actual price).
    """
    last_value = train_val_series.iloc[-1]
    predictions_array = np.full(len(test_series), last_value)
    predictions = pd.Series(predictions_array, index=test_series.index) # Create Series with index
    print(f"Generated Naive Trajectory Forecast (all values = {last_value:.2f}).")
    return predictions

# --- Model Implementation: ARIMA ---
def train_arima(series, order=(1, 1, 1)):
    """
    Trains a standard ARIMA model with a fixed order.
    """
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
        model = ARIMA(series, order=order)
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

def train_auto_arima(series, seasonal=False, **kwargs):
    """
    Trains an ARIMA model by automatically selecting the best order using pmdarima.
    """
    if auto_arima is None:
        print("Error: pmdarima is not installed. Cannot perform Auto ARIMA.")
        return None
    try:
        # pmdarima is generally less strict about frequency but works better with it
        default_kwargs = {
            'start_p': 0, 'start_q': 0, 'max_p': 5, 'max_q': 5,
            'd': None, 'seasonal': seasonal, 'stepwise': True,
            'suppress_warnings': True, 'error_action': 'ignore', 'trace': False
        }
        default_kwargs.update(kwargs)
        # Pass series directly, pmdarima handles numpy/pandas
        model = auto_arima(series, **default_kwargs)
        print(f"Successfully trained Auto ARIMA model. Best order: {model.order}")
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
    """
    predictions = []
    test_index = test_series.index

    print(f"Starting ARIMA Rolling Forecast for {len(test_series)} steps (using fixed model)...")

    is_statsmodels = isinstance(model, ARIMAResultsWrapper)
    is_pmdarima = auto_arima is not None and isinstance(model, pm.arima.ARIMA)

    if not is_statsmodels and not is_pmdarima:
        print("Error: Unknown ARIMA model type provided to arima_rolling_forecast.")
        return None

    # Use a copy for pmdarima as update modifies in-place
    # For statsmodels, extend returns a new object, so we reassign in the loop
    current_model = model
    if is_pmdarima:
        import copy
        current_model = copy.deepcopy(model)

    for t in range(len(test_series)):
        yhat = np.nan
        # Get the actual observation for this step (as a Series with index)
        current_actual_observation = test_series.iloc[t:t+1] # Keep index!

        # Predict step
        try:
            if is_statsmodels:
                yhat = current_model.forecast(steps=1).iloc[0]
            elif is_pmdarima:
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
                 # Extend the model with the new observation (Series with index).
                 # Rely on test_series having the correct frequency from preprocessing.
                 current_model = current_model.extend(current_actual_observation)
            elif is_pmdarima:
                 # Update the pmdarima model state using the Series
                 current_model.update(current_actual_observation)
        except Exception as update_e:
             # Handle update error separately from prediction error
             print(f"Error during model update at step {t}: {type(update_e).__name__} - {update_e}. Forecast quality may degrade.")
             # Model state might be inconsistent for the next step.
             pass # Continue loop, but model state is stale

        if (t + 1) % 50 == 0:
            print(f"Rolling forecast step {t+1}/{len(test_series)} completed.")

    print("ARIMA Rolling Forecast finished.")
    # Ensure the length matches test_series, handle potential NaNs at start/due to errors
    if len(predictions) != len(test_series):
         print(f"Warning: Prediction length ({len(predictions)}) mismatch with test series ({len(test_series)}). Padding with NaNs.")
         # Pad with NaNs at the beginning if prediction is shorter
         predictions = [np.nan] * (len(test_series) - len(predictions)) + predictions

    return pd.Series(predictions, index=test_index)


def arima_trajectory_forecast(train_val_series, test_series, model): # Added test_series for index
    """
    Performs a trajectory forecast using the model's standard multi-step forecast method.
    """
    test_len = len(test_series)
    print(f"Starting ARIMA Trajectory Forecast for {test_len} steps (using multi-step forecast)...")
    is_statsmodels = isinstance(model, ARIMAResultsWrapper)
    is_pmdarima = auto_arima is not None and isinstance(model, pm.arima.ARIMA)

    if is_statsmodels:
        try:
            # Use get_forecast which handles multi-step prediction based on the fitted model
            forecast_result = model.get_forecast(steps=test_len)
            predictions = forecast_result.predicted_mean # This is a Pandas Series
            # Ensure the index matches the test set index
            predictions.index = test_series.index # Assign test index
            print("Statsmodels Trajectory Forecast finished.")
            return predictions
        except Exception as e:
            print(f"Error during statsmodels trajectory forecast: {type(e).__name__} - {e}")
            return None
    elif is_pmdarima:
        try:
            predictions_array = model.predict(n_periods=test_len)
            # Convert numpy array to Pandas Series with the correct index
            predictions = pd.Series(predictions_array, index=test_series.index)
            print("pmdarima Trajectory Forecast finished.")
            return predictions
        except Exception as e:
            print(f"Error during pmdarima trajectory forecast: {type(e).__name__} - {e}")
            return None
    else:
        print("Error: Unknown ARIMA model type provided.")
        return None

# --- Model Implementation: Pure LSTM ---

def build_lstm_sequences(data, window_size):
    """
    Creates sequences and corresponding labels for LSTM training/prediction.
    """
    X, y = [], []
    if len(data) <= window_size: # Check if data is long enough
        return np.array(X), np.array(y)
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def create_lstm_model(window_size, lstm1_units=32, lstm2_units=16, dense_units=16, dropout_rate=0.2, n_features=1):
    """
    Creates the LSTM model architecture as specified in the documentation.
    """
    if tf is None:
        print("Error: tensorflow is not installed. Cannot create LSTM model.")
        return None

    model = Sequential()
    model.add(Input(shape=(window_size, n_features)))
    model.add(LSTM(lstm1_units, return_sequences=True, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm2_units, return_sequences=False, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))

    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    print("LSTM Model Architecture:")
    model.summary()
    return model

def train_lstm(train_series, val_series, window_size, epochs, batch_size, patience, lstm_units_1, lstm_units_2, dense_units, dropout_rate):
    """
    Trains the Pure LSTM model. Includes scaling, sequence building, training with early stopping.
    """
    if tf is None: return None, None, None

    try:
        # Ensure Series have values attribute
        train_vals = train_series.values.reshape(-1, 1)
        val_vals = val_series.values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_vals)
        val_scaled = scaler.transform(val_vals)

        X_train, y_train = build_lstm_sequences(train_scaled.flatten(), window_size)
        X_val, y_val = build_lstm_sequences(val_scaled.flatten(), window_size)

        if X_train.size == 0 or X_val.size == 0:
             print("Error: Not enough data to create sequences for LSTM training/validation.")
             return None, None, None

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        print(f"LSTM Input shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}")

        model = create_lstm_model(window_size, lstm_units_1, lstm_units_2, dense_units, dropout_rate)
        if model is None: return None, None, None

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)

        print("Starting LSTM training...")
        history = model.fit(X_train, y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=(X_val, y_val),
                          callbacks=[early_stopping],
                          verbose=1)

        print("LSTM training finished.")
        return model, scaler, history

    except Exception as e:
        print(f"An error occurred during LSTM training: {e}")
        return None, None, None

def lstm_rolling_forecast(train_val_series, test_series, model, scaler, window_size):
    """
    Performs rolling forecast using a pre-trained LSTM model.
    """
    if tf is None: return None

    try:
        # Combine train_val and test for creating the full history needed for rolling windows
        full_series = pd.concat([train_val_series, test_series])
        # Scale the entire series based on the scaler fitted on training data
        scaled_full_series = scaler.transform(full_series.values.reshape(-1, 1)).flatten()

        predictions_scaled = []
        # Start index in the full scaled series corresponding to the start of the test set
        history_start_index = len(train_val_series)

        print(f"Starting LSTM Rolling Forecast for {len(test_series)} steps...")
        for t in range(len(test_series)):
            # Define the window of actual past data to use for prediction
            current_window_start = history_start_index + t - window_size
            current_window_end = history_start_index + t
            if current_window_start < 0: # Ensure we have enough history
                 print(f"Warning: Not enough history for LSTM window at step {t}. Skipping.")
                 predictions_scaled.append(np.nan)
                 continue

            # Get the input sequence from the scaled full series (actual data)
            input_seq = scaled_full_series[current_window_start:current_window_end]
            input_seq_reshaped = input_seq.reshape((1, window_size, 1))

            # Predict the next step (scaled)
            pred_scaled = model.predict(input_seq_reshaped, verbose=0)[0][0]
            predictions_scaled.append(pred_scaled)

            if (t + 1) % 50 == 0:
                 print(f"Rolling forecast step {t+1}/{len(test_series)} completed.")

        # Inverse transform the scaled predictions
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
        print("LSTM Rolling Forecast finished.")
        # Return as Pandas Series with the test set index
        return pd.Series(predictions, index=test_series.index)

    except Exception as e:
        print(f"An error occurred during LSTM rolling forecast: {e}")
        return None


def lstm_trajectory_forecast(train_val_series, test_series, model, scaler, window_size): # Pass test_series for index
    """
    Performs trajectory forecast using a pre-trained LSTM model.
    """
    if tf is None: return None
    test_len = len(test_series)

    try:
        scaled_train_val = scaler.transform(train_val_series.values.reshape(-1, 1)).flatten()

        if len(scaled_train_val) < window_size:
            print(f"Error: Not enough training data ({len(scaled_train_val)}) for LSTM window ({window_size}).")
            return None

        current_window = list(scaled_train_val[-window_size:])
        predictions_scaled = []

        print(f"Starting LSTM Trajectory Forecast for {test_len} steps...")
        for _ in range(test_len):
            input_seq = np.array(current_window).reshape((1, window_size, 1))
            pred_scaled = model.predict(input_seq, verbose=0)[0][0]
            predictions_scaled.append(pred_scaled)
            # Update window with the prediction
            current_window.pop(0)
            current_window.append(pred_scaled)

        # Inverse transform the scaled predictions
        predictions_array = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
        print("LSTM Trajectory Forecast finished.")
        # Return as Pandas Series with the test set index
        return pd.Series(predictions_array, index=test_series.index)

    except Exception as e:
        print(f"An error occurred during LSTM trajectory forecast: {e}")
        return None

# --- Model Implementation: Hybrid ARIMA + LSTM ---

def calculate_arima_residuals(series, arima_model):
    """
    Calculates the residuals of an ARIMA model on the series it was trained on.
    """
    try:
        if isinstance(arima_model, ARIMAResultsWrapper):
            # Ensure series index matches model's index if possible
            predictions = arima_model.predict(start=series.index[0], end=series.index[-1])
            predictions = predictions.reindex(series.index) # Align index just in case
            residuals = series - predictions
        elif auto_arima is not None and isinstance(arima_model, pm.arima.ARIMA):
            # pmdarima predict_in_sample returns numpy array, needs index
            predictions = arima_model.predict_in_sample()
            residuals = series - pd.Series(predictions, index=series.index)
        else:
            print("Error: Unknown ARIMA model type for residual calculation.")
            return None
        print("Successfully calculated ARIMA residuals.")
        return residuals.dropna()
    except Exception as e:
        print(f"Error calculating ARIMA residuals. Type: {type(e).__name__}")
        return None

# Note: We reuse `train_lstm` for training the residual LSTM.

def hybrid_rolling_forecast(train_val_series, test_series, arima_model, lstm_residual_model, lstm_residual_scaler, window_size):
    """
    Performs rolling forecast using a Hybrid ARIMA+LSTM model.
    """
    if tf is None: return None

    try:
        initial_residuals = calculate_arima_residuals(train_val_series, arima_model)
        if initial_residuals is None:
            print("Failed to calculate initial residuals for hybrid forecast.")
            return None

        # Ensure enough residuals for the first window
        if len(initial_residuals) < window_size:
            print(f"Error: Not enough initial residuals ({len(initial_residuals)}) for LSTM window ({window_size}).")
            return None

        # Use actual residuals from train_val phase for initial LSTM history
        scaled_residuals_history = list(lstm_residual_scaler.transform(initial_residuals.values.reshape(-1, 1)).flatten())
        price_history = list(train_val_series) # For fallback only
        final_predictions = []
        test_index = test_series.index

        is_statsmodels = isinstance(arima_model, ARIMAResultsWrapper)
        is_pmdarima = auto_arima is not None and isinstance(arima_model, pm.arima.ARIMA)

        if not is_statsmodels and not is_pmdarima:
            print("Error: Unknown ARIMA model type provided to hybrid_rolling_forecast.")
            return None

        # Use copies for models that modify state in-place during rolling
        current_arima_model = arima_model
        if is_pmdarima:
            import copy
            current_arima_model = copy.deepcopy(arima_model) # pmdarima update modifies in-place
        elif is_statsmodels:
             # statsmodels extend returns a *new* model object, so we reassign it in the loop
             current_arima_model = arima_model # Start with the original model

        print(f"Starting Hybrid Rolling Forecast for {len(test_series)} steps (using fixed models)...")
        for t in range(len(test_series)):
            arima_pred = np.nan
            # Get the actual observation for this step (as a Series with index)
            current_actual_observation = test_series.iloc[t:t+1] # Keep index!
            current_actual = current_actual_observation.iloc[0] # Get scalar value for price_history list

            # 1. Predict next step with ARIMA (using corrected rolling logic)
            try:
                if is_statsmodels:
                    # Predict based on the model state *before* seeing current_actual
                    arima_pred = current_arima_model.forecast(steps=1).iloc[0]
                elif is_pmdarima:
                    # Use the copied and potentially updated model
                    arima_pred = current_arima_model.predict(n_periods=1)[0]
                else:
                    raise ValueError("Unknown ARIMA model type")
            except Exception as e:
                 print(f"Warning: ARIMA prediction failed at step {t}. Type: {type(e).__name__}. Using fallback (last known price).")
                 arima_pred = price_history[-1] if price_history else 0.0 # Use scalar history for fallback

            # 2. Predict next step residual with LSTM (based on *actual* past residuals)
            lstm_pred_resid = 0.0 # Default residual prediction
            if len(scaled_residuals_history) >= window_size:
                 current_resid_window = scaled_residuals_history[-window_size:]
                 input_seq_resid = np.array(current_resid_window).reshape((1, window_size, 1))
                 try:
                     lstm_pred_scaled = lstm_residual_model.predict(input_seq_resid, verbose=0)[0][0]
                     lstm_pred_resid = lstm_residual_scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
                 except Exception as lstm_e:
                     print(f"Warning: LSTM residual prediction failed at step {t}. Type: {type(lstm_e).__name__}. Using 0.")
                     lstm_pred_resid = 0.0
            else:
                 print(f"Warning: Not enough residual history at step {t}. Using 0 for residual prediction.")

            # 3. Combine predictions
            final_pred = arima_pred + lstm_pred_resid
            final_predictions.append(final_pred)

            # 4. Update histories and ARIMA model state
            price_history.append(current_actual) # Append actual price to external list

            # Calculate actual residual for this step using the ARIMA prediction *for this step*
            actual_residual = current_actual - arima_pred
            try:
                # Scale the actual residual
                scaled_actual_residual = lstm_residual_scaler.transform([[actual_residual]])[0][0]
                scaled_residuals_history.append(scaled_actual_residual) # Append scaled actual residual for next LSTM input
            except Exception as scale_e:
                 print(f"Warning: Failed to scale actual residual at step {t}. Type: {type(scale_e).__name__}. Appending last known or 0.")
                 # Append a fallback value to keep the history length consistent
                 if scaled_residuals_history: scaled_residuals_history.append(scaled_residuals_history[-1])
                 else: scaled_residuals_history.append(0.0)

            # Update ARIMA model state *after* using it for prediction and calculating residual
            try:
                if is_statsmodels:
                    # Extend the model state. extend() returns a new ResultsWrapper object.
                    current_arima_model = current_arima_model.extend(current_actual_observation)
                elif is_pmdarima:
                    # Update the pmdarima model state in-place using the Series
                    current_arima_model.update(current_actual_observation)
            except Exception as update_e:
                 print(f"Warning: ARIMA model update failed at step {t}. Type: {type(update_e).__name__}. Forecast quality may degrade.")
                 # Model state might be inconsistent for the next step

            if (t + 1) % 50 == 0:
                 print(f"Hybrid Rolling forecast step {t+1}/{len(test_series)} completed.")

        print("Hybrid Rolling Forecast finished.")
        # Ensure the length matches test_series
        if len(final_predictions) != len(test_series):
             print(f"Warning: Hybrid prediction length ({len(final_predictions)}) mismatch with test series ({len(test_series)}). Padding with NaNs.")
             final_predictions = [np.nan] * (len(test_series) - len(final_predictions)) + final_predictions
        return pd.Series(final_predictions, index=test_index)

    except Exception as e:
        print(f"An error occurred during Hybrid rolling forecast setup: {e}")
        return None


def hybrid_trajectory_forecast(train_val_series, test_series, arima_model, lstm_residual_model, lstm_residual_scaler, window_size):
    """
    Performs trajectory forecast using a Hybrid ARIMA+LSTM model.
    Uses standard multi-step forecast for ARIMA part and iterative prediction for LSTM residuals.
    """
    if tf is None: return None
    test_len = len(test_series)

    try:
        # 1. Get the full ARIMA trajectory forecast first
        arima_preds = arima_trajectory_forecast(train_val_series, test_series, arima_model)
        if arima_preds is None:
            print("Failed to get ARIMA trajectory for hybrid forecast.")
            return None
        # Ensure arima_preds is a numpy array for easier handling later if needed
        arima_preds_values = arima_preds.values if isinstance(arima_preds, pd.Series) else arima_preds


        # 2. Initialize LSTM residual prediction history
        initial_residuals = calculate_arima_residuals(train_val_series, arima_model) # Actual residuals from training phase
        if initial_residuals is None:
            print("Failed to calculate initial residuals for hybrid trajectory forecast.")
            return None
        scaled_initial_residuals = lstm_residual_scaler.transform(initial_residuals.values.reshape(-1, 1)).flatten()

        if len(scaled_initial_residuals) < window_size:
             print(f"Error: Not enough initial residuals ({len(scaled_initial_residuals)}) to form LSTM window ({window_size}). Cannot start trajectory.")
             return None

        # History window for LSTM residual trajectory (starts with actual scaled residuals)
        scaled_residuals_history_window = list(scaled_initial_residuals[-window_size:])
        final_predictions = []

        print(f"Starting Hybrid Trajectory Forecast for {test_len} steps (ARIMA multi-step + LSTM iterative)...")
        for t in range(test_len):
            # 3. Predict next step residual with LSTM (using predicted residual history)
            lstm_pred_resid = 0.0
            lstm_pred_scaled = 0.0 # Initialize scaled prediction
            try:
                input_seq_resid = np.array(scaled_residuals_history_window).reshape((1, window_size, 1))
                lstm_pred_scaled = lstm_residual_model.predict(input_seq_resid, verbose=0)[0][0]
                lstm_pred_resid = lstm_residual_scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
            except Exception as lstm_e:
                 print(f"Warning: LSTM residual trajectory prediction failed at step {t}. Type: {type(lstm_e).__name__}. Using 0.")
                 lstm_pred_resid = 0.0
                 lstm_pred_scaled = 0.0 # Ensure scaled value is also 0 if prediction fails

            # 4. Combine ARIMA prediction (already computed for all steps) and LSTM residual prediction
            # Ensure arima_preds_values[t] is accessed correctly
            current_arima_pred = arima_preds_values[t] if t < len(arima_preds_values) else np.nan
            final_pred = current_arima_pred + lstm_pred_resid
            final_predictions.append(final_pred)

            # 5. Update LSTM residual history window with the *predicted* scaled residual
            scaled_residuals_history_window.pop(0)
            scaled_residuals_history_window.append(lstm_pred_scaled) # Append predicted scaled residual

            if (t + 1) % 50 == 0:
                 print(f"Hybrid Trajectory forecast step {t+1}/{test_len} completed.")

        print("Hybrid Trajectory Forecast finished.")
        # Return as Pandas Series with correct index
        return pd.Series(final_predictions, index=test_series.index)

    except Exception as e:
        print(f"An error occurred during Hybrid trajectory forecast: {e}")
        return None


# --- Visualization Helper ---
def plot_loss_curves(history, model_name):
    """Plots training and validation loss curves."""
    if history is None or not hasattr(history, 'history') or 'loss' not in history.history or 'val_loss' not in history.history:
        print(f"No history data to plot for {model_name}.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label=f'{model_name} Train Loss')
    plt.plot(history.history['val_loss'], label=f'{model_name} Validation Loss')
    plt.title(f'{model_name} Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    # Save the plot before showing
    plot_filename = f'../results/plots/{model_name}_loss_curve.png'
    try:
        plt.savefig(plot_filename)
        print(f"Loss curve saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving loss curve plot: {e}")
    plt.show()

def plot_predictions(test_series_actual, predictions_dict, title_suffix=""): # Changed first arg
    """Plots actual vs predicted values for multiple models."""
    plt.figure(figsize=(14, 7))
    # Plot actual values using the Series directly (index for x, values for y)
    plt.plot(test_series_actual.index, test_series_actual.values, label='Actual Adj Close', linewidth=2)

    test_dates = test_series_actual.index # Get index for potential alignment

    for model_name, preds in predictions_dict.items():
        if preds is None: # Skip if predictions failed
             print(f"Skipping plot for {model_name} due to missing predictions.")
             continue

        # Handle predictions (Series or ndarray)
        if isinstance(preds, pd.Series):
            valid_preds = preds.dropna()
            if not valid_preds.empty:
                 # Plot Series using its own index
                 plt.plot(valid_preds.index, valid_preds.values, label=f'{model_name} Predicted', linestyle='--')
            else:
                 print(f"Warning: No valid (non-NaN) Series predictions to plot for {model_name}.")
        elif isinstance(preds, np.ndarray):
             # Handle numpy array - plot against test_dates if length matches
             if len(preds) == len(test_dates):
                 # Drop NaNs from array before plotting
                 valid_mask = ~np.isnan(preds)
                 if np.any(valid_mask):
                      plt.plot(test_dates[valid_mask], preds[valid_mask], label=f'{model_name} Predicted (Array)', linestyle=':')
                 else:
                      print(f"Warning: Numpy prediction array contains only NaNs for {model_name}.")
             else:
                 print(f"Warning: Numpy prediction array length mismatch for {model_name}. Cannot plot reliably.")
        else:
             # Fallback for other types (like lists or previously stored .values)
             try:
                 preds_array = np.array(preds)
                 if preds_array.ndim == 1 and len(preds_array) == len(test_dates):
                      valid_mask = ~np.isnan(preds_array)
                      if np.any(valid_mask):
                           plt.plot(test_dates[valid_mask], preds_array[valid_mask], label=f'{model_name} Predicted (Fallback)', linestyle='-.')
                      else:
                           print(f"Warning: Fallback prediction array contains only NaNs for {model_name}.")
                 else:
                      print(f"Warning: Could not interpret or align prediction type ({type(preds)}) for {model_name}. Cannot plot.")
             except Exception as plot_err:
                 print(f"Warning: Error plotting fallback prediction type ({type(preds)}) for {model_name}: {plot_err}")


    plt.title(f'NVDA Stock Price Prediction {title_suffix}')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save the plot before showing
    plot_filename = f'../results/plots/predictions_{title_suffix.lower().replace(" ", "_")}.png'
    try:
        plt.savefig(plot_filename)
        print(f"Prediction plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving prediction plot: {e}")
    plt.show()


# --- Main Execution Area ---
if __name__ == "__main__":
    print("--- NVDA Prediction Analysis Start ---")

    # 1. Load and Preprocess Data
    print("\n[Step 1/7] Loading and Preprocessing Data...")
    df = load_and_preprocess_data(CSV_PATH, CUTOFF_DATE)
    if df is None:
        print("Failed to load/preprocess data. Exiting.")
        exit()

    # 2. Split Data
    print("\n[Step 2/7] Splitting Data...")
    # Pass the DataFrame with DatetimeIndex to split_data
    train_df, val_df, test_df, train_val_df = split_data(df, TRAIN_VAL_RATIO, TRAIN_RATIO)
    if train_df is None:
        print("Failed to split data. Exiting.")
        exit()

    # Prepare series for models (use adj_close) - these will now have DatetimeIndex
    train_series = train_df['adj_close']
    val_series = val_df['adj_close']
    test_series = test_df['adj_close']
    train_val_series = train_val_df['adj_close']

    # Store results
    results = {
        'Rolling': {'Predictions': {}, 'Metrics': {}},
        'Trajectory': {'Predictions': {}, 'Metrics': {}}
    }
    trained_models = {}

    print("\nData loading and splitting complete. Ready for model training and prediction.")
    print("-" * 50)

    # --- 3. Naive Forecast ---
    print("\n[Step 3/7] Running Naive Forecast...")
    model_name_naive = "Naive Forecast"
    naive_preds_rolling = naive_forecast_rolling(test_series)
    # Store Series directly for consistent plotting
    results['Rolling']['Predictions'][model_name_naive] = naive_preds_rolling
    metrics_rolling_naive = evaluate_performance(f"{model_name_naive} (Rolling)", test_series.values, naive_preds_rolling.values)
    results['Rolling']['Metrics'][model_name_naive] = metrics_rolling_naive

    # Pass test_series for index alignment
    naive_preds_trajectory = naive_forecast_trajectory(train_val_series, test_series)
    results['Trajectory']['Predictions'][model_name_naive] = naive_preds_trajectory
    metrics_trajectory_naive = evaluate_performance(f"{model_name_naive} (Trajectory)", test_series.values, naive_preds_trajectory.values)
    results['Trajectory']['Metrics'][model_name_naive] = metrics_trajectory_naive
    print("-" * 50)

    # --- 4. ARIMA(1,1,1) ---
    print("\n[Step 4/7] Running ARIMA(1,1,1)...")
    model_name_arima111 = "ARIMA(1,1,1)"
    # Ensure train_val_series has frequency for statsmodels
    arima111_model = train_arima(train_val_series, order=(1, 1, 1))

    if arima111_model:
        trained_models[model_name_arima111] = arima111_model
        arima111_preds_rolling = arima_rolling_forecast(train_val_series, test_series, arima111_model)
        if arima111_preds_rolling is not None:
            results['Rolling']['Predictions'][model_name_arima111] = arima111_preds_rolling # Store Series
            metrics_rolling = evaluate_performance(f"{model_name_arima111} (Rolling)", test_series.values, arima111_preds_rolling.values)
            results['Rolling']['Metrics'][model_name_arima111] = metrics_rolling
        else:
            print(f"{model_name_arima111} Rolling Forecast failed.")
            results['Rolling']['Metrics'][model_name_arima111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

        # Pass test_series for index alignment
        arima111_preds_trajectory = arima_trajectory_forecast(train_val_series, test_series, arima111_model)
        if arima111_preds_trajectory is not None:
            results['Trajectory']['Predictions'][model_name_arima111] = arima111_preds_trajectory # Store Series
            metrics_trajectory = evaluate_performance(f"{model_name_arima111} (Trajectory)", test_series.values, arima111_preds_trajectory.values)
            results['Trajectory']['Metrics'][model_name_arima111] = metrics_trajectory
        else:
            print(f"{model_name_arima111} Trajectory Forecast failed.")
            results['Trajectory']['Metrics'][model_name_arima111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    else:
        print(f"Skipping {model_name_arima111} due to training failure.")
        results['Rolling']['Metrics'][model_name_arima111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Trajectory']['Metrics'][model_name_arima111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    print("-" * 50)

    # --- 5. Auto ARIMA ---
    print("\n[Step 5/7] Running Auto ARIMA...")
    model_name_auto_arima = "Auto ARIMA"
    # pmdarima usually handles index better, but pass Series just in case
    auto_arima_model = train_auto_arima(train_val_series, seasonal=False,
                                        start_p=0, start_q=0, max_p=5, max_q=5, trace=False)

    if auto_arima_model:
        trained_models[model_name_auto_arima] = auto_arima_model
        # Need a fresh model for rolling forecast as pmdarima update() modifies in-place
        # Retrain on the same data to get an independent object for rolling
        auto_arima_model_for_rolling = train_auto_arima(train_val_series, seasonal=False,
                                                        start_p=0, start_q=0, max_p=5, max_q=5, trace=False)
        if auto_arima_model_for_rolling:
            auto_arima_preds_rolling = arima_rolling_forecast(train_val_series, test_series, auto_arima_model_for_rolling)
            if auto_arima_preds_rolling is not None:
                results['Rolling']['Predictions'][model_name_auto_arima] = auto_arima_preds_rolling # Store Series
                metrics_rolling = evaluate_performance(f"{model_name_auto_arima} (Rolling)", test_series.values, auto_arima_preds_rolling.values)
                results['Rolling']['Metrics'][model_name_auto_arima] = metrics_rolling
            else:
                print(f"{model_name_auto_arima} Rolling Forecast failed.")
                results['Rolling']['Metrics'][model_name_auto_arima] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        else:
             print(f"Failed to get a fresh Auto ARIMA model for rolling forecast.")
             results['Rolling']['Metrics'][model_name_auto_arima] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

        # Pass test_series for index alignment
        auto_arima_preds_trajectory = arima_trajectory_forecast(train_val_series, test_series, auto_arima_model)
        if auto_arima_preds_trajectory is not None:
            results['Trajectory']['Predictions'][model_name_auto_arima] = auto_arima_preds_trajectory # Store Series
            metrics_trajectory = evaluate_performance(f"{model_name_auto_arima} (Trajectory)", test_series.values, auto_arima_preds_trajectory.values)
            results['Trajectory']['Metrics'][model_name_auto_arima] = metrics_trajectory
        else:
            print(f"{model_name_auto_arima} Trajectory Forecast failed.")
            results['Trajectory']['Metrics'][model_name_auto_arima] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    else:
        print(f"Skipping {model_name_auto_arima} due to training failure.")
        results['Rolling']['Metrics'][model_name_auto_arima] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Trajectory']['Metrics'][model_name_auto_arima] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    print("-" * 50)

    # --- 6. Pure LSTM ---
    print("\n[Step 6/7] Running Pure LSTM...")
    model_name_lstm = "Pure LSTM"

    if tf is None:
        print("Skipping Pure LSTM because tensorflow is not installed.")
        results['Rolling']['Metrics'][model_name_lstm] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Trajectory']['Metrics'][model_name_lstm] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    else:
        lstm_model, lstm_scaler, lstm_history = train_lstm(
            train_series, val_series, window_size=LSTM_WINDOW_SIZE, epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE, patience=LSTM_PATIENCE, lstm_units_1=LSTM_UNITS_1,
            lstm_units_2=LSTM_UNITS_2, dense_units=LSTM_DENSE_UNITS, dropout_rate=LSTM_DROPOUT_RATE
        )

        if lstm_model and lstm_scaler: # Check if training was successful
            trained_models[model_name_lstm] = {'model': lstm_model, 'scaler': lstm_scaler}
            if lstm_history: plot_loss_curves(lstm_history, model_name_lstm)

            lstm_preds_rolling = lstm_rolling_forecast(train_val_series, test_series, lstm_model, lstm_scaler, LSTM_WINDOW_SIZE)
            if lstm_preds_rolling is not None:
                results['Rolling']['Predictions'][model_name_lstm] = lstm_preds_rolling # Store Series
                metrics_rolling = evaluate_performance(f"{model_name_lstm} (Rolling)", test_series.values, lstm_preds_rolling.values)
                results['Rolling']['Metrics'][model_name_lstm] = metrics_rolling
            else:
                print(f"{model_name_lstm} Rolling Forecast failed.")
                results['Rolling']['Metrics'][model_name_lstm] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

            # Pass test_series for index alignment
            lstm_preds_trajectory_array = lstm_trajectory_forecast(train_val_series, test_series, lstm_model, lstm_scaler, LSTM_WINDOW_SIZE)
            if lstm_preds_trajectory_array is not None:
                results['Trajectory']['Predictions'][model_name_lstm] = lstm_preds_trajectory_array # Store Series
                metrics_trajectory = evaluate_performance(f"{model_name_lstm} (Trajectory)", test_series.values, lstm_preds_trajectory_array.values)
                results['Trajectory']['Metrics'][model_name_lstm] = metrics_trajectory
            else:
                print(f"{model_name_lstm} Trajectory Forecast failed.")
                results['Trajectory']['Metrics'][model_name_lstm] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        else:
            print(f"Skipping {model_name_lstm} due to training failure.")
            results['Rolling']['Metrics'][model_name_lstm] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
            results['Trajectory']['Metrics'][model_name_lstm] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    print("-" * 50)

    # --- 7. Hybrid Models ---
    print("\n[Step 7/7] Running Hybrid Models...")

    if tf is None:
        print("Skipping Hybrid Models because tensorflow is not installed.")
        results['Rolling']['Metrics']['Hybrid ARIMA(1,1,1)+LSTM'] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Trajectory']['Metrics']['Hybrid ARIMA(1,1,1)+LSTM'] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Rolling']['Metrics']['Hybrid Auto ARIMA+LSTM'] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Trajectory']['Metrics']['Hybrid Auto ARIMA+LSTM'] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    else:
        # --- 7a. Hybrid ARIMA(1,1,1) + LSTM ---
        model_name_hybrid_111 = "Hybrid ARIMA(1,1,1)+LSTM"
        print(f"\n--- Running {model_name_hybrid_111} ---")

        if model_name_arima111 in trained_models:
            base_arima_model_111 = trained_models[model_name_arima111]
            # Need a fresh model instance for rolling forecast if using statsmodels extend
            arima_model_for_hybrid_rolling_111 = train_arima(train_val_series, order=(1, 1, 1))

            if arima_model_for_hybrid_rolling_111: # Check if retraining for rolling worked
                residuals_111 = calculate_arima_residuals(train_val_series, base_arima_model_111)

                if residuals_111 is not None:
                    # Align residuals index with train_val_series for splitting
                    residuals_111 = residuals_111.reindex(train_val_series.index).dropna()
                    # Split residuals based on train/val split indices
                    train_residuals_111 = residuals_111.loc[train_series.index].dropna()
                    val_residuals_111 = residuals_111.loc[val_series.index].dropna()

                    if not train_residuals_111.empty and not val_residuals_111.empty:
                        print(f"Training LSTM for {model_name_hybrid_111} residuals...")
                        lstm_resid_111_model, lstm_resid_111_scaler, lstm_resid_111_history = train_lstm(
                            train_residuals_111, val_residuals_111, window_size=LSTM_WINDOW_SIZE, epochs=LSTM_EPOCHS,
                            batch_size=LSTM_BATCH_SIZE, patience=LSTM_PATIENCE, lstm_units_1=LSTM_UNITS_1,
                            lstm_units_2=LSTM_UNITS_2, dense_units=LSTM_DENSE_UNITS, dropout_rate=LSTM_DROPOUT_RATE
                        )

                        if lstm_resid_111_model and lstm_resid_111_scaler:
                            trained_models[model_name_hybrid_111] = {'arima': base_arima_model_111, 'lstm': lstm_resid_111_model, 'scaler': lstm_resid_111_scaler}
                            if lstm_resid_111_history: plot_loss_curves(lstm_resid_111_history, f"{model_name_hybrid_111} Residual LSTM")

                            # Rolling Forecast - Use the fresh ARIMA model instance
                            hybrid_111_preds_rolling = hybrid_rolling_forecast(train_val_series, test_series, arima_model_for_hybrid_rolling_111, lstm_resid_111_model, lstm_resid_111_scaler, LSTM_WINDOW_SIZE)
                            if hybrid_111_preds_rolling is not None:
                                results['Rolling']['Predictions'][model_name_hybrid_111] = hybrid_111_preds_rolling # Store Series
                                metrics_rolling = evaluate_performance(f"{model_name_hybrid_111} (Rolling)", test_series.values, hybrid_111_preds_rolling.values)
                                results['Rolling']['Metrics'][model_name_hybrid_111] = metrics_rolling
                            else:
                                print(f"{model_name_hybrid_111} Rolling Forecast failed.")
                                results['Rolling']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

                            # Trajectory Forecast - Use the original base ARIMA model
                            hybrid_111_preds_trajectory = hybrid_trajectory_forecast(train_val_series, test_series, base_arima_model_111, lstm_resid_111_model, lstm_resid_111_scaler, LSTM_WINDOW_SIZE)
                            if hybrid_111_preds_trajectory is not None:
                                results['Trajectory']['Predictions'][model_name_hybrid_111] = hybrid_111_preds_trajectory # Store Series
                                metrics_trajectory = evaluate_performance(f"{model_name_hybrid_111} (Trajectory)", test_series.values, hybrid_111_preds_trajectory.values)
                                results['Trajectory']['Metrics'][model_name_hybrid_111] = metrics_trajectory
                            else:
                                print(f"{model_name_hybrid_111} Trajectory Forecast failed.")
                                results['Trajectory']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                        else:
                            print(f"Skipping {model_name_hybrid_111} forecasts due to LSTM residual training failure.")
                            results['Rolling']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                            results['Trajectory']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                    else:
                        print(f"Skipping {model_name_hybrid_111} forecasts due to insufficient residual data after splitting.")
                        results['Rolling']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                        results['Trajectory']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                else:
                    print(f"Skipping {model_name_hybrid_111} forecasts due to residual calculation failure.")
                    results['Rolling']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                    results['Trajectory']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
            else:
                 print(f"Skipping {model_name_hybrid_111} rolling forecast because retraining ARIMA(1,1,1) failed.")
                 results['Rolling']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                 # Still attempt trajectory with the base model
                 if base_arima_model_111 and 'lstm' in trained_models.get(model_name_hybrid_111, {}): # Check if LSTM model exists
                      lstm_resid_111_model = trained_models[model_name_hybrid_111]['lstm']
                      lstm_resid_111_scaler = trained_models[model_name_hybrid_111]['scaler']
                      hybrid_111_preds_trajectory = hybrid_trajectory_forecast(train_val_series, test_series, base_arima_model_111, lstm_resid_111_model, lstm_resid_111_scaler, LSTM_WINDOW_SIZE)
                      if hybrid_111_preds_trajectory is not None:
                           results['Trajectory']['Predictions'][model_name_hybrid_111] = hybrid_111_preds_trajectory # Store Series
                           metrics_trajectory = evaluate_performance(f"{model_name_hybrid_111} (Trajectory)", test_series.values, hybrid_111_preds_trajectory.values)
                           results['Trajectory']['Metrics'][model_name_hybrid_111] = metrics_trajectory
                      else:
                           print(f"{model_name_hybrid_111} Trajectory Forecast failed.")
                           results['Trajectory']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                 else:
                      print(f"Skipping {model_name_hybrid_111} trajectory forecast as base ARIMA or LSTM model is missing.")
                      results['Trajectory']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

        else:
            print(f"Skipping {model_name_hybrid_111} because base ARIMA(1,1,1) model training failed.")
            results['Rolling']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
            results['Trajectory']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

        # --- 7b. Hybrid Auto ARIMA + LSTM ---
        model_name_hybrid_auto = "Hybrid Auto ARIMA+LSTM"
        print(f"\n--- Running {model_name_hybrid_auto} ---")

        if model_name_auto_arima in trained_models:
            base_auto_arima_model = trained_models[model_name_auto_arima]
            # Need a fresh model for rolling forecast as pmdarima update() modifies in-place
            auto_arima_model_for_hybrid_rolling = train_auto_arima(train_val_series, seasonal=False,
                                                                    start_p=0, start_q=0, max_p=5, max_q=5, trace=False) # Retrain

            if auto_arima_model_for_hybrid_rolling: # Check if retraining for rolling worked
                residuals_auto = calculate_arima_residuals(train_val_series, base_auto_arima_model)

                if residuals_auto is not None:
                    residuals_auto = residuals_auto.reindex(train_val_series.index).dropna()
                    train_residuals_auto = residuals_auto.loc[train_series.index].dropna()
                    val_residuals_auto = residuals_auto.loc[val_series.index].dropna()

                    if not train_residuals_auto.empty and not val_residuals_auto.empty:
                        print(f"Training LSTM for {model_name_hybrid_auto} residuals...")
                        lstm_resid_auto_model, lstm_resid_auto_scaler, lstm_resid_auto_history = train_lstm(
                            train_residuals_auto, val_residuals_auto, window_size=LSTM_WINDOW_SIZE, epochs=LSTM_EPOCHS,
                            batch_size=LSTM_BATCH_SIZE, patience=LSTM_PATIENCE, lstm_units_1=LSTM_UNITS_1,
                            lstm_units_2=LSTM_UNITS_2, dense_units=LSTM_DENSE_UNITS, dropout_rate=LSTM_DROPOUT_RATE
                        )

                        if lstm_resid_auto_model and lstm_resid_auto_scaler:
                            trained_models[model_name_hybrid_auto] = {'arima': base_auto_arima_model, 'lstm': lstm_resid_auto_model, 'scaler': lstm_resid_auto_scaler}
                            if lstm_resid_auto_history: plot_loss_curves(lstm_resid_auto_history, f"{model_name_hybrid_auto} Residual LSTM")

                            # Rolling Forecast - Use the fresh Auto ARIMA model instance
                            hybrid_auto_preds_rolling = hybrid_rolling_forecast(train_val_series, test_series, auto_arima_model_for_hybrid_rolling, lstm_resid_auto_model, lstm_resid_auto_scaler, LSTM_WINDOW_SIZE)
                            if hybrid_auto_preds_rolling is not None:
                                results['Rolling']['Predictions'][model_name_hybrid_auto] = hybrid_auto_preds_rolling # Store Series
                                metrics_rolling = evaluate_performance(f"{model_name_hybrid_auto} (Rolling)", test_series.values, hybrid_auto_preds_rolling.values)
                                results['Rolling']['Metrics'][model_name_hybrid_auto] = metrics_rolling
                            else:
                                print(f"{model_name_hybrid_auto} Rolling Forecast failed.")
                                results['Rolling']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

                            # Trajectory Forecast (use original trained Auto ARIMA) - Pass test_series
                            hybrid_auto_preds_trajectory = hybrid_trajectory_forecast(train_val_series, test_series, base_auto_arima_model, lstm_resid_auto_model, lstm_resid_auto_scaler, LSTM_WINDOW_SIZE)
                            if hybrid_auto_preds_trajectory is not None:
                                results['Trajectory']['Predictions'][model_name_hybrid_auto] = hybrid_auto_preds_trajectory # Store Series
                                metrics_trajectory = evaluate_performance(f"{model_name_hybrid_auto} (Trajectory)", test_series.values, hybrid_auto_preds_trajectory.values)
                                results['Trajectory']['Metrics'][model_name_hybrid_auto] = metrics_trajectory
                            else:
                                print(f"{model_name_hybrid_auto} Trajectory Forecast failed.")
                                results['Trajectory']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                        else:
                            print(f"Skipping {model_name_hybrid_auto} forecasts due to LSTM residual training failure.")
                            results['Rolling']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                            results['Trajectory']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                    else:
                        print(f"Skipping {model_name_hybrid_auto} forecasts due to insufficient residual data after splitting.")
                        results['Rolling']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                        results['Trajectory']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                else:
                    print(f"Skipping {model_name_hybrid_auto} forecasts due to residual calculation failure.")
                    results['Rolling']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                    results['Trajectory']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
            else:
                 print(f"Skipping {model_name_hybrid_auto} rolling forecast because retraining Auto ARIMA failed.")
                 results['Rolling']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                 # Still attempt trajectory with the base model
                 if base_auto_arima_model and 'lstm' in trained_models.get(model_name_hybrid_auto, {}): # Check if LSTM model exists
                      lstm_resid_auto_model = trained_models[model_name_hybrid_auto]['lstm']
                      lstm_resid_auto_scaler = trained_models[model_name_hybrid_auto]['scaler']
                      hybrid_auto_preds_trajectory = hybrid_trajectory_forecast(train_val_series, test_series, base_auto_arima_model, lstm_resid_auto_model, lstm_resid_auto_scaler, LSTM_WINDOW_SIZE)
                      if hybrid_auto_preds_trajectory is not None:
                           results['Trajectory']['Predictions'][model_name_hybrid_auto] = hybrid_auto_preds_trajectory # Store Series
                           metrics_trajectory = evaluate_performance(f"{model_name_hybrid_auto} (Trajectory)", test_series.values, hybrid_auto_preds_trajectory.values)
                           results['Trajectory']['Metrics'][model_name_hybrid_auto] = metrics_trajectory
                      else:
                           print(f"{model_name_hybrid_auto} Trajectory Forecast failed.")
                           results['Trajectory']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                 else:
                      print(f"Skipping {model_name_hybrid_auto} trajectory forecast as base ARIMA or LSTM model is missing.")
                      results['Trajectory']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

        else:
            print(f"Skipping {model_name_hybrid_auto} because base Auto ARIMA model training failed.")
            results['Rolling']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
            results['Trajectory']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

    print("-" * 50)
    print("\nHybrid Models complete.")

    # --- Final Results Aggregation and Visualization ---
    print("\n--- Aggregating and Visualizing Results ---")

    # Prepare data for plotting - Use the test_series directly
    # test_dates = test_df.index # No longer needed here, index is in test_series
    # y_true = test_series.values # No longer needed here, pass test_series

    # Plot Rolling Forecasts
    plot_predictions(test_series, results['Rolling']['Predictions'], title_suffix="Rolling Forecast")

    # Plot Trajectory Forecasts
    plot_predictions(test_series, results['Trajectory']['Predictions'], title_suffix="Trajectory Forecast")

    # Display Metrics Tables
    rolling_metrics_df = pd.DataFrame(results['Rolling']['Metrics']).T
    trajectory_metrics_df = pd.DataFrame(results['Trajectory']['Metrics']).T

    print("\n--- Rolling Forecast Metrics ---")
    # print(rolling_metrics_df.to_markdown(floatfmt=".4f")) # Requires tabulate
    print(rolling_metrics_df)


    print("\n--- Trajectory Forecast Metrics ---")
    # print(trajectory_metrics_df.to_markdown(floatfmt=".4f")) # Requires tabulate
    print(trajectory_metrics_df)

    print("\n--- NVDA Prediction Analysis End ---")

