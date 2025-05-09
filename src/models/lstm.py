# -*- coding: utf-8 -*-
"""
Implementation of Pure LSTM model and related functions.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings

# Handle TensorFlow import separately
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, TensorBoard # Import TensorBoard
    from tensorflow.keras.optimizers import Adam
    import datetime # Import datetime for log directories
    import os # Import os for path joining
    TF_AVAILABLE = True
    # GPU Memory Growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs (LSTM Module)")
        except RuntimeError as e:
            print(f"GPU Memory growth error (LSTM Module): {e}")
except ImportError:
    print("Warning: tensorflow not found. LSTM functionality will be unavailable.")
    TF_AVAILABLE = False
    # Define dummy classes/functions if TF not available to avoid NameErrors later
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None
    Input = None
    EarlyStopping = None
    Adam = None


# Import config for default parameters
try:
    from .. import config # Relative import from parent directory (src)
except ImportError:
    import config # Fallback for direct execution

def build_lstm_sequences(data, window_size=config.LSTM_WINDOW_SIZE):
    """
    Creates sequences and corresponding labels for LSTM training/prediction.

    Args:
        data (np.ndarray): Scaled time series data (1D array).
        window_size (int): The number of time steps in each input sequence. Defaults to config.LSTM_WINDOW_SIZE.

    Returns:
        tuple: Contains np.ndarray X (sequences) and np.ndarray y (labels).
    """
    X, y = [], []
    if len(data) <= window_size: # Check if data is long enough
        print("[LSTM Model] Warning: Not enough data to build sequences.")
        return np.array(X), np.array(y)
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def create_lstm_model(window_size, lstm1_units, lstm2_units, dense_units, dropout_rate,
                      activation='tanh', use_recurrent_dropout=True, bidirectional=False, n_features=1):
    """
    Creates the LSTM or Bi-LSTM model architecture based on provided parameters.

    Args:
        window_size (int): Input sequence length.
        lstm1_units (int): Units in the first LSTM layer.
        lstm2_units (int): Units in the second LSTM layer.
        dense_units (int): Units in the intermediate Dense layer.
        dropout_rate (float): Dropout rate.
        activation (str): Activation function for LSTM layers ('tanh' or 'relu'). Defaults to 'tanh'.
        use_recurrent_dropout (bool): If True, use recurrent_dropout in LSTM layers.
                                      If False, use standalone Dropout layers after LSTM layers. Defaults to True.
        bidirectional (bool): If True, use Bidirectional LSTM layers. Defaults to False.
        n_features (int): Number of input features (usually 1 for univariate). Defaults to 1.

    Returns:
        tf.keras.models.Sequential or None: The compiled Keras model, or None if TensorFlow not available.
    """
    if not TF_AVAILABLE:
        print("[LSTM Model] Error: tensorflow is not installed. Cannot create LSTM model.")
        return None

    model_type = "Bi-LSTM" if bidirectional else "LSTM"
    print(f"\n[LSTM Model] Creating {model_type} Model Architecture (Activation: {activation}, Recurrent Dropout: {use_recurrent_dropout})...")
    model = Sequential()
    model.add(Input(shape=(window_size, n_features)))

    # First LSTM/Bi-LSTM Layer
    lstm1 = LSTM(lstm1_units, return_sequences=True, activation=activation, recurrent_dropout=dropout_rate if use_recurrent_dropout else 0.0)
    if bidirectional:
        model.add(Bidirectional(lstm1)) # return_sequences=True is implicit from inner LSTM
    else:
        model.add(lstm1)
    if not use_recurrent_dropout and not bidirectional: # Standalone dropout only makes sense for unidirectional here
         model.add(Dropout(dropout_rate))

    # Second LSTM/Bi-LSTM Layer
    # IMPORTANT: return_sequences=False for the last recurrent layer before Dense
    lstm2 = LSTM(lstm2_units, return_sequences=False, activation=activation, recurrent_dropout=dropout_rate if use_recurrent_dropout else 0.0)
    if bidirectional:
        model.add(Bidirectional(lstm2)) # return_sequences=False is implicit
    else:
        model.add(lstm2)
    if not use_recurrent_dropout and not bidirectional:
         model.add(Dropout(dropout_rate))

    # Dense Layers
    model.add(Dense(dense_units, activation='sigmoid')) # Changed Dense layer activation to sigmoid
    model.add(Dense(1)) # Output layer

    optimizer = Adam() # Consider adding learning_rate parameter here if needed later
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    print("LSTM Model Architecture Summary:")
    model.summary()
    return model

def train_lstm(train_series, val_series,
               # Explicitly list all parameters, remove defaults linking to config
               window_size,
               epochs,
               batch_size,
               patience,
               lstm_units_1,
               lstm_units_2,
               dense_units,
               dropout_rate,
               activation='tanh', # Default activation for LSTM layers
               use_recurrent_dropout=True, # Default dropout style
               bidirectional=False, # Default model type
               model_name="LSTM_Run"): # Added model_name for logging
    """
    Trains the LSTM or Bi-LSTM model. Includes scaling, sequence building, training with early stopping,
    and TensorBoard logging. Accepts specific hyperparameters.

    Args:
        train_series (pd.Series): Training time series data (prices or residuals).
        val_series (pd.Series): Validation time series data (prices or residuals).
        window_size (int): Input sequence length.
        epochs (int): Max number of training epochs.
        batch_size (int): Training batch size.
        patience (int): Early stopping patience.
        lstm_units_1 (int): Units in the first LSTM layer.
        lstm_units_2 (int): Units in the second LSTM layer.
        dense_units (int): Units in the intermediate Dense layer.
        dropout_rate (float): Dropout rate.
        activation (str): Activation function for LSTM layers ('tanh' or 'relu').
        use_recurrent_dropout (bool): Whether to use recurrent_dropout or standalone Dropout layers.
        bidirectional (bool): Whether to create a Bidirectional LSTM model.
        model_name (str): A name for this training run, used for TensorBoard log directory.

    Returns:
        tuple: (model, scaler, history) or (None, None, None) if training fails or TF not available.
               model: Trained Keras model.
               scaler: Fitted MinMaxScaler.
               history: Keras training history object.
    """
    if not TF_AVAILABLE:
        print("[LSTM Model] Error: tensorflow is not installed. Cannot train LSTM model.")
        return None, None, None

    print(f"\n[LSTM Model] Starting LSTM training process (Epochs: {epochs}, Patience: {patience}, Batch: {batch_size})...")
    try:
        # Ensure Series have values attribute and correct shape for scaler
        train_vals = train_series.values.reshape(-1, 1)
        val_vals = val_series.values.reshape(-1, 1)

        # Using MinMaxScaler as per original spec, though StandardScaler might be alternative
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_vals)
        val_scaled = scaler.transform(val_vals)
        print("Data scaled using MinMaxScaler(0,1).")

        X_train, y_train = build_lstm_sequences(train_scaled.flatten(), window_size)
        X_val, y_val = build_lstm_sequences(val_scaled.flatten(), window_size)

        if X_train.size == 0 or X_val.size == 0:
             print("Error: Not enough data to create sequences for LSTM training/validation after building sequences.")
             return None, None, None

        # Reshape for LSTM input [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        print(f"LSTM Input shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}")

        # Pass specific parameters to create_lstm_model
        model = create_lstm_model(
            window_size=window_size,
            lstm1_units=lstm_units_1,
            lstm2_units=lstm_units_2,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            activation=activation,
            use_recurrent_dropout=use_recurrent_dropout,
            bidirectional=bidirectional # Pass bidirectional flag
        )
        if model is None: return None, None, None

        # --- TensorBoard Setup ---
        # Create a unique log directory for this run
        run_log_dir = os.path.join(config.LOGS_DIR, f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        print(f"[TensorBoard] Logging to: {run_log_dir}")
        tensorboard_callback = TensorBoard(
            log_dir=run_log_dir,
            histogram_freq=1,      # Log histograms every epoch
            write_graph=True,      # Log the model graph
            write_images=False,    # We don't have images to log
            update_freq='epoch',   # Log metrics every epoch
            profile_batch=0        # Disable profiler for now (can enable later if needed)
        )
        # --- End TensorBoard Setup ---

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)

        print("Starting Keras model fitting...")
        # Add tensorboard_callback to the list of callbacks
        history = model.fit(X_train, y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=(X_val, y_val),
                          callbacks=[early_stopping, tensorboard_callback], # Added TensorBoard callback
                          verbose=1) # Set verbose=1 to see progress

        print("[LSTM Model] LSTM training finished.")
        return model, scaler, history

    except Exception as e:
        print(f"An error occurred during LSTM training: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        return None, None, None

def lstm_rolling_forecast(train_val_series, test_series, model, scaler, window_size=config.LSTM_WINDOW_SIZE):
    """
    Performs rolling forecast using a pre-trained LSTM model.

    Args:
        train_val_series (pd.Series): Training and validation data.
        test_series (pd.Series): The test time series data.
        model (tf.keras.models.Sequential): The pre-trained LSTM model.
        scaler (MinMaxScaler): The scaler fitted on the training data.
        window_size (int): Input sequence length. Defaults to config.LSTM_WINDOW_SIZE.

    Returns:
        pd.Series or None: The rolling forecast predictions, or None if forecasting fails.
    """
    if not TF_AVAILABLE or model is None or scaler is None:
        print("[LSTM Model] Error: Missing prerequisites (TF, model, or scaler) for rolling forecast.")
        return None

    print(f"\n[LSTM Model] Starting LSTM Rolling Forecast for {len(test_series)} steps...")
    try:
        # Combine train_val and test for creating the full history needed for rolling windows
        full_series = pd.concat([train_val_series, test_series])
        # Scale the entire series based on the scaler fitted on training data
        # Ensure correct shape for scaler
        scaled_full_series = scaler.transform(full_series.values.reshape(-1, 1)).flatten()

        predictions_scaled = []
        # Start index in the full scaled series corresponding to the start of the test set
        history_start_index = len(train_val_series)

        for t in range(len(test_series)):
            # Define the window of actual past data to use for prediction
            current_window_start = history_start_index + t - window_size
            current_window_end = history_start_index + t
            if current_window_start < 0: # Ensure we have enough history
                 print(f"Warning: Not enough history for LSTM window at step {t}. Appending NaN.")
                 predictions_scaled.append(np.nan) # Append NaN if not enough history
                 continue

            # Get the input sequence from the scaled full series (actual data)
            input_seq = scaled_full_series[current_window_start:current_window_end]
            # Reshape for LSTM input [samples, time steps, features]
            input_seq_reshaped = input_seq.reshape((1, window_size, 1))

            # Predict the next step (scaled)
            try:
                pred_scaled = model.predict(input_seq_reshaped, verbose=0)[0][0]
                predictions_scaled.append(pred_scaled)
            except Exception as pred_e:
                 print(f"Error during LSTM prediction at step {t}: {pred_e}. Appending NaN.")
                 predictions_scaled.append(np.nan)


            if (t + 1) % 50 == 0 or t == len(test_series) - 1:
                 print(f"Rolling forecast step {t+1}/{len(test_series)} completed.")

        # Inverse transform the scaled predictions
        # Handle potential NaNs before inverse transform
        predictions_scaled_array = np.array(predictions_scaled) # Keep as 1D
        nan_mask = np.isnan(predictions_scaled_array)
        valid_predictions_scaled = predictions_scaled_array[~nan_mask].reshape(-1, 1) # Reshape valid data for scaler

        # Initialize predictions as a 1D array
        predictions = np.full(len(test_series), np.nan, dtype=float)

        if len(valid_predictions_scaled) > 0:
             # Inverse transform valid data and assign back using the mask
             predictions[~nan_mask] = scaler.inverse_transform(valid_predictions_scaled).flatten()

        print("[LSTM Model] LSTM Rolling Forecast finished.")
        # Return as Pandas Series with the test set index
        return pd.Series(predictions, index=test_series.index) # Already 1D

    except Exception as e:
        print(f"An error occurred during LSTM rolling forecast: {e}")
        import traceback
        traceback.print_exc()
        return None


def lstm_trajectory_forecast(train_val_series, test_series, model, scaler, window_size=config.LSTM_WINDOW_SIZE):
    """
    Performs trajectory forecast using a pre-trained LSTM model.

    Args:
        train_val_series (pd.Series): Training and validation data.
        test_series (pd.Series): The test time series data (used for length and index).
        model (tf.keras.models.Sequential): The pre-trained LSTM model.
        scaler (MinMaxScaler): The scaler fitted on the training data.
        window_size (int): Input sequence length. Defaults to config.LSTM_WINDOW_SIZE.

    Returns:
        pd.Series or None: The trajectory forecast predictions, or None if forecasting fails.
    """
    if not TF_AVAILABLE or model is None or scaler is None:
        print("[LSTM Model] Error: Missing prerequisites (TF, model, or scaler) for trajectory forecast.")
        return None

    test_len = len(test_series)
    print(f"\n[LSTM Model] Starting LSTM Trajectory Forecast for {test_len} steps...")

    try:
        # Scale the training data
        scaled_train_val = scaler.transform(train_val_series.values.reshape(-1, 1)).flatten()

        if len(scaled_train_val) < window_size:
            print(f"Error: Not enough training data ({len(scaled_train_val)}) for LSTM window ({window_size}).")
            return None

        # Initialize the prediction window with the last 'window_size' elements of scaled training data
        current_window = list(scaled_train_val[-window_size:])
        predictions_scaled = []

        for i in range(test_len):
            # Reshape window for LSTM input
            input_seq = np.array(current_window).reshape((1, window_size, 1))

            # Predict the next step (scaled)
            try:
                pred_scaled = model.predict(input_seq, verbose=0)[0][0]
            except Exception as pred_e:
                print(f"Error during LSTM trajectory prediction at step {i}: {pred_e}. Appending NaN and stopping.")
                # Append NaNs for the rest of the forecast if prediction fails
                predictions_scaled.extend([np.nan] * (test_len - i))
                break # Stop the loop

            predictions_scaled.append(pred_scaled)

            # Update window: remove the first element and append the prediction
            current_window.pop(0)
            current_window.append(pred_scaled)

            if (i + 1) % 50 == 0 or i == test_len - 1:
                 print(f"Trajectory forecast step {i+1}/{test_len} completed.")


        # Inverse transform the scaled predictions
        predictions_scaled_array = np.array(predictions_scaled) # Keep as 1D
        # Handle potential NaNs before inverse transform
        nan_mask = np.isnan(predictions_scaled_array)
        valid_predictions_scaled = predictions_scaled_array[~nan_mask].reshape(-1, 1) # Reshape valid data for scaler

        # Initialize predictions as a 1D array
        predictions_array = np.full(len(test_series), np.nan, dtype=float)

        if len(valid_predictions_scaled) > 0:
             # Inverse transform valid data and assign back using the mask
             predictions_array[~nan_mask] = scaler.inverse_transform(valid_predictions_scaled).flatten()

        print("[LSTM Model] LSTM Trajectory Forecast finished.")
        # Return as Pandas Series with the test set index
        return pd.Series(predictions_array, index=test_series.index) # Already 1D

    except Exception as e:
        print(f"An error occurred during LSTM trajectory forecast: {e}")
        import traceback
        traceback.print_exc()
        return None