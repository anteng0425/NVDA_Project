# -*- coding: utf-8 -*-
"""
Implementation of Pure LSTM model and related functions.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings

# Import config first as it's needed for device setup before TensorFlow fully initializes
try:
    from .. import config
except ImportError:
    import config # Fallback for direct execution

# Handle TensorFlow import separately
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
    from tensorflow.keras.optimizers import Adam
    import datetime # Import datetime for log directories
    import os # Import os for path joining
    
    TF_AVAILABLE = True

    # --- Apply Training Device Configuration ---
    if config.FORCE_CPU_TRAINING: # Use the directly imported 'config'
        try:
            tf.config.set_visible_devices([], 'GPU')
            logical_gpus_after_disable = tf.config.list_logical_devices('GPU')
            if not logical_gpus_after_disable:
                 print("[Device Control] GPUs successfully hidden. TensorFlow will use CPU.")
            else:
                 print("[Device Control] Attempted to hide GPUs, but some are still visible. Check TensorFlow initialization order.")
        except Exception as e:
            print(f"[Device Control] Error trying to hide GPUs. They might already be initialized or no GPUs present. Error: {e}")
            print("[Device Control] Proceeding, TensorFlow will use available devices (CPU or any visible GPUs).")
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(f"[Device Control] {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs configured for memory growth (LSTM Module).")
            except RuntimeError as e:
                print(f"[Device Control] GPU Memory growth error (LSTM Module): {e}")
        else:
            print("[Device Control] No GPUs detected by TensorFlow. Using CPU.")

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
    # If TF is not available, config might still be needed for other parts if this module were extended
    # but for now, the primary use of config here is with TF.
    # Ensure config is defined even if TF import fails, if it's used outside the TF_AVAILABLE block.
    if 'config' not in locals(): # If the top-level import failed and TF failed
        try:
            from .. import config
        except ImportError:
            try:
                import config
            except ImportError:
                print("Critical Error: config module could not be imported in lstm.py")
                # Or raise an error, or define a dummy config object if parts of the file can run without it
                pass # Allow script to continue if config is not strictly needed when TF is False

# config is now guaranteed to be imported (or attempted) once at the top for the rest of the module.

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

# The following N-Step Teacher Forcing LSTM model was an experiment to improve trajectory prediction.
# However, it resulted in poor performance, with predictions appearing as a significantly
# time-shifted version of the actual trajectory. Due to these issues, these configurations are commented out.
# def build_lstm_sequences_for_n_step_tf(data, window_size=config.LSTM_WINDOW_SIZE, n_steps_ahead=config.N_STEP_AHEAD_TF):
#     """
#     Creates sequences and corresponding multi-step-ahead labels for LSTM N-Step Teacher Forcing training.
#
#     Args:
#         data (np.ndarray): Scaled time series data (1D array).
#         window_size (int): The number of time steps in each input sequence.
#         n_steps_ahead (int): The number of future steps to predict and use as labels.
#
#     Returns:
#         tuple: Contains np.ndarray X (input sequences) and np.ndarray Y_n_steps (multi-step target sequences).
#                Returns empty arrays if not enough data.
#     """
#     X, Y_n_steps = [], []
#     # Ensure there's enough data for at least one X sequence and its n_steps_ahead Y sequence
#     if len(data) < window_size + n_steps_ahead:
#         print("[LSTM Model] Warning: Not enough data to build sequences for N-Step TF.")
#         return np.array(X), np.array(Y_n_steps)
#
#     for i in range(len(data) - window_size - n_steps_ahead + 1):
#         X.append(data[i:(i + window_size)])
#         Y_n_steps.append(data[(i + window_size):(i + window_size + n_steps_ahead)])
#    
#     if not X: # If X is empty after the loop (e.g., due to very short data or large n_steps_ahead)
#         print("[LSTM Model] Warning: No sequences generated for N-Step TF, possibly due to data length vs window/n_steps.")
#         return np.array(X), np.array(Y_n_steps)
#        
#     return np.array(X), np.array(Y_n_steps)

def create_lstm_model(window_size, lstm1_units, lstm2_units, dense_units,
                      standard_dropout_rate, recurrent_dropout_rate, # Updated dropout params
                      activation='tanh', use_recurrent_dropout=True, bidirectional=False, n_features=1):
    """
    Creates the LSTM or Bi-LSTM model architecture based on provided parameters.

    Args:
        window_size (int): Input sequence length.
        lstm1_units (int): Units in the first LSTM layer.
        lstm2_units (int): Units in the second LSTM layer.
        dense_units (int): Units in the intermediate Dense layer.
        standard_dropout_rate (float): Dropout rate for non-recurrent connections.
        recurrent_dropout_rate (float): Dropout rate for recurrent connections.
        activation (str): Activation function for LSTM layers ('tanh' or 'relu'). Defaults to 'tanh'.
        use_recurrent_dropout (bool): If True, apply recurrent_dropout_rate to recurrent connections.
        bidirectional (bool): If True, use Bidirectional LSTM layers. Defaults to False.
        n_features (int): Number of input features (usually 1 for univariate). Defaults to 1.

    Returns:
        tf.keras.models.Sequential or None: The compiled Keras model, or None if TensorFlow not available.
    """
    if not TF_AVAILABLE:
        print("[LSTM Model] Error: tensorflow is not installed. Cannot create LSTM model.")
        return None

    model_type = "Bi-LSTM" if bidirectional else "LSTM"
    print(f"\n[LSTM Model] Creating {model_type} Model Architecture (Activation: {activation}, Standard Dropout: {standard_dropout_rate}, Recurrent Dropout: {recurrent_dropout_rate if use_recurrent_dropout else 0.0})...")
    model = Sequential()
    model.add(Input(shape=(window_size, n_features)))

    # First LSTM/Bi-LSTM Layer
    lstm1 = LSTM(lstm1_units,
                 return_sequences=True,
                 activation=activation,
                 dropout=standard_dropout_rate, # Apply standard dropout to non-recurrent connections
                 recurrent_dropout=recurrent_dropout_rate if use_recurrent_dropout else 0.0) # Apply recurrent dropout
    if bidirectional:
        model.add(Bidirectional(lstm1))
    else:
        model.add(lstm1)
    # No separate Dropout layer here as per user preference

    # Second LSTM/Bi-LSTM Layer
    # IMPORTANT: return_sequences=False for the last recurrent layer before Dense
    lstm2 = LSTM(lstm2_units,
                 return_sequences=False,
                 activation=activation,
                 dropout=standard_dropout_rate, # Apply standard dropout
                 recurrent_dropout=recurrent_dropout_rate if use_recurrent_dropout else 0.0) # Apply recurrent dropout
    if bidirectional:
        model.add(Bidirectional(lstm2))
    else:
        model.add(lstm2)
    # No separate Dropout layer here

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
               standard_dropout_rate=config.LSTM_STANDARD_DROPOUT_RATE, # Default from config
               recurrent_dropout_rate=config.LSTM_RECURRENT_DROPOUT_RATE, # Default from config
               activation='tanh', # Default activation for LSTM layers
               use_recurrent_dropout=True, # Default dropout style
               bidirectional=False, # Default model type
               model_name="LSTM_Run", # Added model_name for logging
               apply_scaling: bool = True, # New parameter to control scaling
               scaler_feature_range: tuple = (0, 1)): # New parameter for scaler range
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
        standard_dropout_rate (float): Dropout rate for non-recurrent connections. Defaults to config.LSTM_STANDARD_DROPOUT_RATE.
        recurrent_dropout_rate (float): Dropout rate for recurrent connections. Defaults to config.LSTM_RECURRENT_DROPOUT_RATE.
        activation (str): Activation function for LSTM layers ('tanh' or 'relu').
        use_recurrent_dropout (bool): Whether to apply recurrent_dropout_rate to recurrent connections.
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

       scaler = None # Initialize scaler
       if apply_scaling:
           # Using MinMaxScaler, allow feature_range to be specified
           scaler = MinMaxScaler(feature_range=scaler_feature_range)
           train_scaled_np = scaler.fit_transform(train_vals)
           val_scaled_np = scaler.transform(val_vals)
           print("Data scaled using MinMaxScaler(0,1).") # Note: This print might be misleading if range is (-1,1)
           train_scaled = train_scaled_np.flatten()
           val_scaled = val_scaled_np.flatten()
       else:
           train_scaled = train_vals.flatten() # Use original values, ensure 1D
           val_scaled = val_vals.flatten()   # Use original values, ensure 1D
           print("Data scaling skipped for LSTM training.")

       X_train, y_train = build_lstm_sequences(train_scaled, window_size) # train_scaled is already 1D
       X_val, y_val = build_lstm_sequences(val_scaled, window_size)     # val_scaled is already 1D

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
           standard_dropout_rate=standard_dropout_rate, # Pass new param
           recurrent_dropout_rate=recurrent_dropout_rate, # Pass new param
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

# The following N-Step Teacher Forcing LSTM model was an experiment to improve trajectory prediction.
# However, it resulted in poor performance, with predictions appearing as a significantly
# time-shifted version of the actual trajectory. Due to these issues, these configurations are commented out.
# def train_lstm_n_step_teacher_forcing(
#     train_series, val_series,
#     window_size, epochs, batch_size, patience,
#     lstm_units_1, lstm_units_2, dense_units, dropout_rate,
#     activation='tanh', use_recurrent_dropout=True, bidirectional=False,
#     model_name="LSTM_NStepTF_Run", n_steps_ahead=config.N_STEP_AHEAD_TF,
#     learning_rate=0.001 # Added learning rate parameter
# ):
#     """
#     Trains the LSTM or Bi-LSTM model using N-Step Ahead Teacher Forcing with a custom training loop.
#     The model architecture itself still predicts one step at a time (Dense(1)).
#     The custom loop simulates n-step prediction with teacher forcing for loss calculation.
#
#     Args:
#         train_series (pd.Series): Training time series data.
#         val_series (pd.Series): Validation time series data.
#         window_size (int): Input sequence length.
#         epochs (int): Max number of training epochs.
#         batch_size (int): Training batch size.
#         patience (int): Early stopping patience.
#         lstm_units_1 (int): Units in the first LSTM layer.
#         lstm_units_2 (int): Units in the second LSTM layer.
#         dense_units (int): Units in the intermediate Dense layer.
#         dropout_rate (float): Dropout rate.
#         activation (str): Activation function for LSTM layers.
#         use_recurrent_dropout (bool): Whether to use recurrent_dropout.
#         bidirectional (bool): Whether to create a Bidirectional LSTM model.
#         model_name (str): Name for TensorBoard logging.
#         n_steps_ahead (int): Number of future steps to predict and use for loss calculation.
#         learning_rate (float): Learning rate for the Adam optimizer.
#
#     Returns:
#         tuple: (model, scaler, history_log) or (None, None, None) if training fails or TF not available.
#                model: Trained Keras model.
#                scaler: Fitted MinMaxScaler.
#                history_log: Dict containing lists of train_loss and val_loss per epoch.
#     """
#     if not TF_AVAILABLE:
#         print("[LSTM N-Step TF Model] Error: tensorflow is not installed. Cannot train model.")
#         return None, None, None
#
#     print(f"\n[LSTM N-Step TF Model] Starting training (Epochs: {epochs}, Patience: {patience}, Batch: {batch_size}, N-Steps: {n_steps_ahead})...")
#     try:
#         train_vals = train_series.values.reshape(-1, 1)
#         val_vals = val_series.values.reshape(-1, 1)
#
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         train_scaled = scaler.fit_transform(train_vals).flatten() # Flatten for sequence building
#         val_scaled = scaler.transform(val_vals).flatten()
#         print("Data scaled using MinMaxScaler(0,1).")
#
#         # Use the new sequence builder
#         X_train, Y_train_n_steps = build_lstm_sequences_for_n_step_tf(train_scaled, window_size, n_steps_ahead)
#         X_val, Y_val_n_steps = build_lstm_sequences_for_n_step_tf(val_scaled, window_size, n_steps_ahead)
#
#         if X_train.size == 0 or X_val.size == 0:
#             print("Error: Not enough data to create sequences for N-Step TF training/validation.")
#             return None, None, None
#
#         # Reshape X for LSTM input [samples, time steps, features]
#         X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
#         X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
#        
#         # X_train, X_val, Y_train_n_steps, Y_val_n_steps are initially numpy arrays (likely float64)
#         # Type casting will be handled more locally where needed for TF operations.
#
#         print(f"N-Step TF Input shapes: X_train={X_train.shape}, Y_train_n_steps={Y_train_n_steps.shape}")
#         print(f"N-Step TF Validation shapes: X_val={X_val.shape}, Y_val_n_steps={Y_val_n_steps.shape}")
#
#         model = create_lstm_model(
#             window_size=window_size,
#             lstm1_units=lstm_units_1,
#             lstm2_units=lstm_units_2,
#             dense_units=dense_units,
#             dropout_rate=dropout_rate,
#             activation=activation,
#             use_recurrent_dropout=use_recurrent_dropout,
#             bidirectional=bidirectional
#         )
#         if model is None: return None, None, None
#
#         optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#         loss_fn = tf.keras.losses.MeanSquaredError() # Use standard MSE
#
#         # --- TensorBoard Setup ---
#         run_log_dir = os.path.join(config.LOGS_DIR, f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
#         train_summary_writer = tf.summary.create_file_writer(os.path.join(run_log_dir, 'train'))
#         val_summary_writer = tf.summary.create_file_writer(os.path.join(run_log_dir, 'validation'))
#         print(f"[TensorBoard - N-Step TF] Logging to: {run_log_dir}")
#         # Log model graph (optional, can be large)
#         # tf.summary.trace_on(graph=True, profiler=False) # Start tracing for graph
#         # dummy_input = tf.random.uniform((1, window_size, 1)) # Create a dummy input
#         # _ = model(dummy_input) # Forward pass to build the graph
#         # with train_summary_writer.as_default():
#         #    tf.summary.trace_export(name=f"{model_name}_graph", step=0, profiler_outdir=None)
#         # tf.summary.trace_off()
#
#
#         # --- Custom Training Loop ---
#         history_log = {'train_loss': [], 'val_loss': []}
#         best_val_loss = float('inf')
#         patience_counter = 0
#
#         # Ensure data is tf.float32 before creating dataset
#         X_train_tf = tf.cast(X_train, dtype=tf.float32)
#         Y_train_n_steps_tf = tf.cast(Y_train_n_steps, dtype=tf.float32)
#         X_val_tf = tf.cast(X_val, dtype=tf.float32)
#         Y_val_n_steps_tf = tf.cast(Y_val_n_steps, dtype=tf.float32)
#
#         train_dataset = tf.data.Dataset.from_tensor_slices((X_train_tf, Y_train_n_steps_tf)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
#         val_dataset = tf.data.Dataset.from_tensor_slices((X_val_tf, Y_val_n_steps_tf)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
#
#         @tf.function
#         def process_batch(x_batch_input, y_batch_true_n_steps_input, training):
#             # x_batch_input shape: (batch_size, window_size, 1)
#             # y_batch_true_n_steps_input shape: (batch_size, n_steps_ahead)
#            
#             batch_size_actual = tf.shape(x_batch_input)[0]
#            
#             # Manual unrolling for n_steps_ahead = 5
#             # Ensure n_steps_ahead is indeed 5 for this unrolled version, or add assertion
#             # tf.debugging.assert_equal(tf.constant(n_steps_ahead), tf.constant(5),
#             #                            message="This unrolled version of process_batch expects n_steps_ahead=5")
#
#             predictions_list = [] # Python list to collect step predictions for the batch
#
#             # Step 1
#             current_input_step1 = x_batch_input
#             pred_step1 = model(current_input_step1, training=training) # Shape: (batch_size, 1)
#             predictions_list.append(pred_step1)
#             true_vals_step1 = y_batch_true_n_steps_input[:, 0] # Shape: (batch_size,)
#             true_vals_step1_reshaped = tf.reshape(true_vals_step1, [batch_size_actual, 1, 1])
#             current_input_step2 = tf.concat([current_input_step1[:, 1:, :], true_vals_step1_reshaped], axis=1)
#
#             # Step 2
#             pred_step2 = model(current_input_step2, training=training)
#             predictions_list.append(pred_step2)
#             true_vals_step2 = y_batch_true_n_steps_input[:, 1]
#             true_vals_step2_reshaped = tf.reshape(true_vals_step2, [batch_size_actual, 1, 1])
#             current_input_step3 = tf.concat([current_input_step2[:, 1:, :], true_vals_step2_reshaped], axis=1)
#
#             # Step 3
#             pred_step3 = model(current_input_step3, training=training)
#             predictions_list.append(pred_step3)
#             true_vals_step3 = y_batch_true_n_steps_input[:, 2]
#             true_vals_step3_reshaped = tf.reshape(true_vals_step3, [batch_size_actual, 1, 1])
#             current_input_step4 = tf.concat([current_input_step3[:, 1:, :], true_vals_step3_reshaped], axis=1)
#
#             # Step 4
#             pred_step4 = model(current_input_step4, training=training)
#             predictions_list.append(pred_step4)
#             true_vals_step4 = y_batch_true_n_steps_input[:, 3]
#             true_vals_step4_reshaped = tf.reshape(true_vals_step4, [batch_size_actual, 1, 1])
#             current_input_step5 = tf.concat([current_input_step4[:, 1:, :], true_vals_step4_reshaped], axis=1)
#            
#             # Step 5 (Final prediction for the sequence of 5)
#             pred_step5 = model(current_input_step5, training=training)
#             predictions_list.append(pred_step5)
#            
#             # predictions_list contains 5 tensors, each of shape (batch_size, 1)
#             # Stack along a new dimension (axis=1 for (batch_size, n_steps, 1) or axis=0 for (n_steps, batch_size, 1))
#             # Let's make it (batch_size, n_steps_ahead, 1)
#             stacked_predictions_BSU = tf.stack(predictions_list, axis=1) # Shape: (batch_size, 5, 1)
#             squeezed_predictions = tf.squeeze(stacked_predictions_BSU, axis=-1) # Shape: (batch_size, 5)
#            
#             batch_loss = loss_fn(y_batch_true_n_steps_input, squeezed_predictions)
#             return batch_loss, squeezed_predictions
#
#         @tf.function
#         def train_step_vectorized(x_batch_input, y_batch_true_n_steps_input):
#             with tf.GradientTape() as tape:
#                 avg_batch_loss_step, _ = process_batch(x_batch_input, y_batch_true_n_steps_input, training=True)
#             gradients_step = tape.gradient(avg_batch_loss_step, model.trainable_variables)
#             optimizer.apply_gradients(zip(gradients_step, model.trainable_variables))
#             return avg_batch_loss_step
#
#         @tf.function
#         def val_step_vectorized(x_batch_input_val, y_batch_true_n_steps_input_val):
#             avg_batch_val_loss_step, _ = process_batch(x_batch_input_val, y_batch_true_n_steps_input_val, training=False)
#             return avg_batch_val_loss_step
#
#         for epoch in range(epochs):
#             print(f"Epoch {epoch+1}/{epochs}")
#             epoch_train_loss_avg = tf.keras.metrics.Mean()
#             batch_idx = 0
#             for x_batch, y_batch_true_n_steps in train_dataset:
#                 avg_batch_loss = train_step_vectorized(x_batch, y_batch_true_n_steps)
#                 epoch_train_loss_avg.update_state(avg_batch_loss)
#                 if batch_idx % 20 == 0: # Print progress every 20 batches
#                     print(f"  Train Batch {batch_idx+1}/{tf.data.experimental.cardinality(train_dataset).numpy()}, Avg Batch Loss: {avg_batch_loss.numpy():.4f}")
#                 batch_idx +=1
#            
#             current_train_loss = epoch_train_loss_avg.result().numpy()
#             history_log['train_loss'].append(current_train_loss)
#             with train_summary_writer.as_default():
#                 tf.summary.scalar('epoch_loss', current_train_loss, step=epoch)
#
#             epoch_val_loss_avg = tf.keras.metrics.Mean()
#             batch_idx_val = 0
#             for x_batch_val, y_batch_true_n_steps_val in val_dataset:
#                 avg_batch_val_loss = val_step_vectorized(x_batch_val, y_batch_true_n_steps_val)
#                 epoch_val_loss_avg.update_state(avg_batch_val_loss)
#                 if batch_idx_val % 20 == 0: # Print progress every 20 batches
#                      print(f"  Val Batch {batch_idx_val+1}/{tf.data.experimental.cardinality(val_dataset).numpy()}, Avg Val Batch Loss: {avg_batch_val_loss.numpy():.4f}")
#                 batch_idx_val +=1
#
#             current_val_loss = epoch_val_loss_avg.result().numpy()
#             history_log['val_loss'].append(current_val_loss)
#             with val_summary_writer.as_default():
#                 tf.summary.scalar('epoch_loss', current_val_loss, step=epoch)
#            
#             print(f"Epoch {epoch+1} - Training Loss: {current_train_loss:.4f}, Validation Loss: {current_val_loss:.4f}")
#
#             if current_val_loss < best_val_loss:
#                 best_val_loss = current_val_loss
#                 patience_counter = 0
#                 # Optionally save best model weights here
#                 # model.save_weights(os.path.join(run_log_dir, 'best_model_weights.h5')) # Consider saving
#                 print(f"Validation loss improved to {best_val_loss:.4f}. Resetting patience.")
#             else:
#                 patience_counter += 1
#                 print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
#            
#             if patience_counter >= patience:
#                 print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
#                 break
#        
#         print("[LSTM N-Step TF Model] Training finished.")
#         # Load best weights if saved
#         # if os.path.exists(os.path.join(run_log_dir, 'best_model_weights.h5')):
#         #     print("Loading best model weights.")
#         #     model.load_weights(os.path.join(run_log_dir, 'best_model_weights.h5'))
#            
#         return model, scaler, history_log
#
#     except Exception as e:
#         print(f"An error occurred during LSTM N-Step TF training: {e}")
#         import traceback
#         traceback.print_exc()
#         return None, None, None


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
    if not TF_AVAILABLE or model is None: # Removed scaler is None check
        print("[LSTM Model] Error: Missing prerequisites (TF or model) for rolling forecast.")
        return None

    print(f"\n[LSTM Model] Starting LSTM Rolling Forecast for {len(test_series)} steps...")
    try:
        # Combine train_val and test for creating the full history needed for rolling windows
        full_series = pd.concat([train_val_series, test_series])
        # Scale the entire series based on the scaler fitted on training data
        # Ensure correct shape for scaler
        if scaler is not None:
            scaled_full_series = scaler.transform(full_series.values.reshape(-1, 1)).flatten()
        else:
            scaled_full_series = full_series.values.flatten() # Use original values

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
            if scaler is not None:
                # Inverse transform valid data and assign back using the mask
                predictions[~nan_mask] = scaler.inverse_transform(valid_predictions_scaled).flatten()
            else:
                # Already in original scale
                predictions[~nan_mask] = valid_predictions_scaled.flatten()

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
    if not TF_AVAILABLE or model is None: # Removed scaler is None check
        print("[LSTM Model] Error: Missing prerequisites (TF or model) for trajectory forecast.")
        return None

    test_len = len(test_series)
    print(f"\n[LSTM Model] Starting LSTM Trajectory Forecast for {test_len} steps...")

    try:
        # Scale the training data
        if scaler is not None:
            scaled_train_val = scaler.transform(train_val_series.values.reshape(-1, 1)).flatten()
        else:
            scaled_train_val = train_val_series.values.flatten() # Use original values

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
            if scaler is not None:
                # Inverse transform valid data and assign back using the mask
                predictions_array[~nan_mask] = scaler.inverse_transform(valid_predictions_scaled).flatten()
            else:
                # Already in original scale
                predictions_array[~nan_mask] = valid_predictions_scaled.flatten()

        print("[LSTM Model] LSTM Trajectory Forecast finished.")
        # Return as Pandas Series with the test set index
        return pd.Series(predictions_array, index=test_series.index) # Already 1D

    except Exception as e:
        print(f"An error occurred during LSTM trajectory forecast: {e}")
        import traceback
        traceback.print_exc()
        return None