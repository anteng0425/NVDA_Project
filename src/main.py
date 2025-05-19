# -*- coding: utf-8 -*-
"""
Main script to run the NVDA stock prediction analysis pipeline.
"""

# Standard Libraries
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Keep for potential direct use or plt.show() control
import os

# --- Import Project Modules ---
# Use relative imports assuming execution with `python -m src.main`
from . import config
from .data_processing import load_and_preprocess_data, split_data
from .evaluation import evaluate_performance
from .visualization import plot_loss_curves, plot_predictions, plot_full_history
from .models import naive, arima, lstm, hybrid
from .models.lstm import TF_AVAILABLE
# --- Import New Modules ---
from . import data_processing_iceemdan # Keep relative import
from .models import seq2seq_attention_lstm # Keep relative import

# --- Configuration ---
warnings.filterwarnings("ignore") # Ignore harmless warnings
# Plot style is applied in visualization.py
# Import TensorFlow specific components if TF is available
if TF_AVAILABLE:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
    import datetime # For TensorBoard log directory naming

# --- Main Execution Area ---
if __name__ == "__main__":
    print("--- NVDA Prediction Analysis Start ---")
    # Ensure results directory exists (done in config.py, but safe to re-check)
    os.makedirs(config.RESULTS_PLOTS_DIR, exist_ok=True) # Use updated variable name

    # 1. Load and Preprocess Data
    print("\n[Step 1/7] Loading and Preprocessing Data...")
    # Uses paths and cutoff date from config module
    df = load_and_preprocess_data()
    if df is None:
        print("Failed to load/preprocess data. Exiting.")
        exit() # Correct indentation for exit()

    # --- Plot Full Historical Data ---
    plot_full_history(df) # Call the new plotting function

    # 2. Split Data
    print("\n[Step 2/7] Splitting Data...")
    # Uses ratios from config module
    train_df, val_df, test_df, train_val_df = split_data(df)
    if train_df is None:
        print("Failed to split data. Exiting.")
        exit()

    # Prepare series for models - Create both original and log-transformed versions
    print("\n[Data Prep] Preparing original and log-transformed series...")

    # Original scale series (for Pure LSTM, evaluation, plotting)
    train_series_orig = train_df['adj_close'].copy()
    val_series_orig = val_df['adj_close'].copy()
    test_series_orig = test_df['adj_close'].copy() # Used for y_true in evaluation/plotting
    train_val_series_orig = train_val_df['adj_close'].copy()
    print(f"[Data Prep] Original test_series mean: {test_series_orig.mean():.2f}")

    # Log-transformed series (for Naive, ARIMA, Hybrid models)
    # Handle potential non-positive values before log transform if necessary
    if (train_series_orig <= 0).any() or (val_series_orig <= 0).any() or (test_series_orig <= 0).any() or (train_val_series_orig <= 0).any():
        print("Error: Non-positive values detected in 'adj_close' before log transform. Cannot proceed.")
        exit()
    train_series_log = np.log(train_series_orig)
    val_series_log = np.log(val_series_orig)
    test_series_log = np.log(test_series_orig) # Log version for model prediction structure (index/length)
    train_val_series_log = np.log(train_val_series_orig)
    print(f"[Data Prep] Log-transformed train_series mean: {train_series_log.mean():.2f}")


    # Store results - predictions will be stored in original scale
    results = {
        'Rolling': {'Predictions': {}, 'Metrics': {}},
        'Trajectory': {'Predictions': {}, 'Metrics': {}}
    }
    trained_models = {} # Store trained models for potential reuse (e.g., in hybrid)

    print("\nData loading and splitting complete. Ready for model training and prediction.")
    print("-" * 50)

    # --- 3. Naive Forecast ---
    print("\n[Step 3/7] Running Naive Forecast...")
    model_name_naive = "Naive Forecast"
    try:
        # Naive models operate on the log-transformed series, predictions are exponentiated
        naive_preds_rolling_log = naive.naive_forecast_rolling(test_series_log)
        naive_preds_rolling_final = np.exp(naive_preds_rolling_log) if naive_preds_rolling_log is not None else None
        results['Rolling']['Predictions'][model_name_naive] = naive_preds_rolling_final
        metrics_rolling_naive = evaluate_performance(f"{model_name_naive} (Rolling)", test_series_orig.values, naive_preds_rolling_final.values if naive_preds_rolling_final is not None else None)
        results['Rolling']['Metrics'][model_name_naive] = metrics_rolling_naive

        naive_preds_trajectory_log = naive.naive_forecast_trajectory(train_val_series_log, test_series_log)
        naive_preds_trajectory_final = np.exp(naive_preds_trajectory_log) if naive_preds_trajectory_log is not None else None
        results['Trajectory']['Predictions'][model_name_naive] = naive_preds_trajectory_final
        metrics_trajectory_naive = evaluate_performance(f"{model_name_naive} (Trajectory)", test_series_orig.values, naive_preds_trajectory_final.values if naive_preds_trajectory_final is not None else None)
        results['Trajectory']['Metrics'][model_name_naive] = metrics_trajectory_naive
    except Exception as e:
        print(f"Error during Naive Forecast: {e}")
        results['Rolling']['Metrics'][model_name_naive] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Trajectory']['Metrics'][model_name_naive] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    print("-" * 50)

    # --- 4. ARIMA(1,1,1) ---
    print("\n[Step 4/7] Running ARIMA(1,1,1)...")
    model_name_arima111 = "ARIMA(1,1,1)"
    arima111_model = None # Initialize
    arima111_preds_rolling_log = None # Initialize here for broader scope
    try:
        # ARIMA models operate on the log-transformed series, predictions are exponentiated
        arima111_model = arima.train_arima(train_val_series_log, order=(1, 1, 1))

        if arima111_model:
            trained_models[model_name_arima111] = arima111_model
            # Rolling Forecast
            arima111_preds_rolling_log = arima.arima_rolling_forecast(train_val_series_log, test_series_log, arima111_model)
            arima111_preds_rolling_final = np.exp(arima111_preds_rolling_log) if arima111_preds_rolling_log is not None else None
            if arima111_preds_rolling_final is not None:
                results['Rolling']['Predictions'][model_name_arima111] = arima111_preds_rolling_final
                metrics_rolling = evaluate_performance(f"{model_name_arima111} (Rolling)", test_series_orig.values, arima111_preds_rolling_final.values)
                results['Rolling']['Metrics'][model_name_arima111] = metrics_rolling
            else:
                print(f"{model_name_arima111} Rolling Forecast failed.")
                results['Rolling']['Metrics'][model_name_arima111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

            # Trajectory Forecast
            arima111_preds_trajectory_log = arima.arima_trajectory_forecast(train_val_series_log, test_series_log, arima111_model)
            arima111_preds_trajectory_final = np.exp(arima111_preds_trajectory_log) if arima111_preds_trajectory_log is not None else None
            if arima111_preds_trajectory_final is not None:
                results['Trajectory']['Predictions'][model_name_arima111] = arima111_preds_trajectory_final
                metrics_trajectory = evaluate_performance(f"{model_name_arima111} (Trajectory)", test_series_orig.values, arima111_preds_trajectory_final.values)
                results['Trajectory']['Metrics'][model_name_arima111] = metrics_trajectory
            else:
                print(f"{model_name_arima111} Trajectory Forecast failed.")
                results['Trajectory']['Metrics'][model_name_arima111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        else:
            print(f"Skipping {model_name_arima111} forecasts due to training failure.")
            results['Rolling']['Metrics'][model_name_arima111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
            results['Trajectory']['Metrics'][model_name_arima111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    except Exception as e:
        print(f"Error during ARIMA(1,1,1) processing: {e}")
        results['Rolling']['Metrics'][model_name_arima111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Trajectory']['Metrics'][model_name_arima111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    print("-" * 50)


    # --- 5. Auto ARIMA ---
    print("\n[Step 5/7] Running Auto ARIMA...")
    model_name_auto_arima = "Auto ARIMA"
    auto_arima_model = None # Initialize
    try:
        # Auto ARIMA operates on the log-transformed series, predictions are exponentiated
        auto_arima_model = arima.train_auto_arima(train_val_series_log)

        if auto_arima_model:
            trained_models[model_name_auto_arima] = auto_arima_model
            # Need a fresh model for rolling forecast as pmdarima update() modifies in-place
            auto_arima_model_for_rolling = arima.train_auto_arima(train_val_series_log) # Retrain

            if auto_arima_model_for_rolling:
                auto_arima_preds_rolling_log = arima.arima_rolling_forecast(train_val_series_log, test_series_log, auto_arima_model_for_rolling)
                auto_arima_preds_rolling_final = np.exp(auto_arima_preds_rolling_log) if auto_arima_preds_rolling_log is not None else None
                if auto_arima_preds_rolling_final is not None:
                    results['Rolling']['Predictions'][model_name_auto_arima] = auto_arima_preds_rolling_final
                    metrics_rolling = evaluate_performance(f"{model_name_auto_arima} (Rolling)", test_series_orig.values, auto_arima_preds_rolling_final.values)
                    results['Rolling']['Metrics'][model_name_auto_arima] = metrics_rolling
                else:
                    print(f"{model_name_auto_arima} Rolling Forecast failed.")
                    results['Rolling']['Metrics'][model_name_auto_arima] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
            else:
                 print(f"Failed to get a fresh Auto ARIMA model for rolling forecast.")
                 results['Rolling']['Metrics'][model_name_auto_arima] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

            # Trajectory Forecast (use original trained model)
            auto_arima_preds_trajectory_log = arima.arima_trajectory_forecast(train_val_series_log, test_series_log, auto_arima_model)
            auto_arima_preds_trajectory_final = np.exp(auto_arima_preds_trajectory_log) if auto_arima_preds_trajectory_log is not None else None
            if auto_arima_preds_trajectory_final is not None:
                results['Trajectory']['Predictions'][model_name_auto_arima] = auto_arima_preds_trajectory_final
                metrics_trajectory = evaluate_performance(f"{model_name_auto_arima} (Trajectory)", test_series_orig.values, auto_arima_preds_trajectory_final.values)
                results['Trajectory']['Metrics'][model_name_auto_arima] = metrics_trajectory
            else:
                print(f"{model_name_auto_arima} Trajectory Forecast failed.")
                results['Trajectory']['Metrics'][model_name_auto_arima] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        else:
            print(f"Skipping {model_name_auto_arima} forecasts due to training failure.")
            results['Rolling']['Metrics'][model_name_auto_arima] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
            results['Trajectory']['Metrics'][model_name_auto_arima] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    except Exception as e:
        print(f"Error during Auto ARIMA processing: {e}")
        results['Rolling']['Metrics'][model_name_auto_arima] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Trajectory']['Metrics'][model_name_auto_arima] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    print("-" * 50)


    # --- 6. Pure LSTM ---
    print("\n[Step 6/7] Running Pure LSTM...")
    model_name_lstm = "Pure LSTM"
    lstm_model, lstm_scaler, lstm_history = None, None, None # Initialize
    try:
        if not TF_AVAILABLE:
            print("Skipping Pure LSTM because tensorflow is not installed.")
            results['Rolling']['Metrics'][model_name_lstm] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
            results['Trajectory']['Metrics'][model_name_lstm] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        else:
            # Pure LSTM operates directly on original prices
            # The scaler inside train_lstm will be fitted on original price data.
            # Predictions from lstm_..._forecast will be on original scale.
            lstm_model, lstm_scaler, lstm_history = lstm.train_lstm(
                train_series=train_series_orig, # Pass original prices
                val_series=val_series_orig,
                # Pass Pure LSTM specific parameters from config
                window_size=config.LSTM_WINDOW_SIZE,
                epochs=config.PURE_LSTM_EPOCHS,
                batch_size=config.PURE_LSTM_BATCH_SIZE, # Now 128 again
                patience=config.PURE_LSTM_PATIENCE,
                lstm_units_1=config.PURE_LSTM_UNITS_1,
                lstm_units_2=config.PURE_LSTM_UNITS_2,
                dense_units=config.PURE_LSTM_DENSE_UNITS,
                standard_dropout_rate=config.LSTM_STANDARD_DROPOUT_RATE, # From shared config
                recurrent_dropout_rate=config.LSTM_RECURRENT_DROPOUT_RATE, # From shared config
                activation='tanh', # Explicitly set for Pure LSTM
                use_recurrent_dropout=True, # Ensure recurrent_dropout is applied for Pure LSTM
                bidirectional=True, # Use Bi-LSTM for Pure model
                model_name="Pure_BiLSTM" # Added model name for logging
            )

            if lstm_model and lstm_scaler: # Check if training was successful
                trained_models[model_name_lstm] = {'model': lstm_model, 'scaler': lstm_scaler} # Scaler is for original prices
                if lstm_history: plot_loss_curves(lstm_history, model_name_lstm) # Loss is on scaled original prices

                # Rolling Forecast - Pass original series, result is original scale
                lstm_preds_rolling_final = lstm.lstm_rolling_forecast(train_val_series_orig, test_series_orig, lstm_model, lstm_scaler)
                # NO np.exp() needed
                if lstm_preds_rolling_final is not None:
                    results['Rolling']['Predictions'][model_name_lstm] = lstm_preds_rolling_final
                    metrics_rolling = evaluate_performance(f"{model_name_lstm} (Rolling)", test_series_orig.values, lstm_preds_rolling_final.values)
                    results['Rolling']['Metrics'][model_name_lstm] = metrics_rolling
                else:
                    print(f"{model_name_lstm} Rolling Forecast failed.")
                    results['Rolling']['Metrics'][model_name_lstm] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

                # Trajectory Forecast - Pass original series, result is original scale
                lstm_preds_trajectory_final = lstm.lstm_trajectory_forecast(train_val_series_orig, test_series_orig, lstm_model, lstm_scaler)
                # NO np.exp() needed
                if lstm_preds_trajectory_final is not None:
                    results['Trajectory']['Predictions'][model_name_lstm] = lstm_preds_trajectory_final
                    metrics_trajectory = evaluate_performance(f"{model_name_lstm} (Trajectory)", test_series_orig.values, lstm_preds_trajectory_final.values)
                    results['Trajectory']['Metrics'][model_name_lstm] = metrics_trajectory
                else:
                    print(f"{model_name_lstm} Trajectory Forecast failed.")
                    results['Trajectory']['Metrics'][model_name_lstm] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
            else:
                print(f"Skipping {model_name_lstm} forecasts due to training failure.")
                results['Rolling']['Metrics'][model_name_lstm] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                results['Trajectory']['Metrics'][model_name_lstm] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    except Exception as e:
        print(f"Error during Pure LSTM processing: {e}")
        results['Rolling']['Metrics'][model_name_lstm] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Trajectory']['Metrics'][model_name_lstm] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    print("-" * 50)

# The following N-Step Teacher Forcing LSTM model was an experiment to improve trajectory prediction.
# However, it resulted in poor performance, with predictions appearing as a significantly
# time-shifted version of the actual trajectory. Due to these issues, these configurations are commented out.
    # # --- New Step: Pure LSTM N-Step Teacher Forcing ---
    # print(f"\n[Step 7/8] Running Pure LSTM N-Step TF (N={config.N_STEP_AHEAD_TF})...")
    # model_name_lstm_nstep_tf = f"Pure LSTM N-Step TF (N={config.N_STEP_AHEAD_TF})"
    # lstm_nstep_tf_model, lstm_nstep_tf_scaler, lstm_nstep_tf_history_log = None, None, None

    # try:
    #     if not TF_AVAILABLE:
    #         print(f"Skipping {model_name_lstm_nstep_tf} because tensorflow is not installed.")
    #         results['Rolling']['Metrics'][model_name_lstm_nstep_tf] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    #         results['Trajectory']['Metrics'][model_name_lstm_nstep_tf] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    #     else:
    #         lstm_nstep_tf_model, lstm_nstep_tf_scaler, lstm_nstep_tf_history_log = train_lstm_n_step_teacher_forcing(
    #             train_series=train_series_orig,
    #             val_series=val_series_orig,
    #             window_size=config.LSTM_WINDOW_SIZE, # Shared window size
    #             epochs=config.PURE_LSTM_NSTEP_TF_EPOCHS,
    #             batch_size=config.PURE_LSTM_NSTEP_TF_BATCH_SIZE,
    #             patience=config.PURE_LSTM_NSTEP_TF_PATIENCE,
    #             lstm_units_1=config.PURE_LSTM_NSTEP_TF_UNITS_1,
    #             lstm_units_2=config.PURE_LSTM_NSTEP_TF_UNITS_2,
    #             dense_units=config.PURE_LSTM_NSTEP_TF_DENSE_UNITS,
    #             dropout_rate=config.LSTM_DROPOUT_RATE, # Shared dropout rate
    #             activation='tanh', # Consistent with Pure LSTM
    #             use_recurrent_dropout=True, # Consistent with Pure LSTM
    #             bidirectional=True, # Consistent with Pure LSTM
    #             model_name=f"Pure_LSTM_NStep_TF_{config.N_STEP_AHEAD_TF}",
    #             n_steps_ahead=config.N_STEP_AHEAD_TF
    #         )

    #         if lstm_nstep_tf_model and lstm_nstep_tf_scaler:
    #             trained_models[model_name_lstm_nstep_tf] = {'model': lstm_nstep_tf_model, 'scaler': lstm_nstep_tf_scaler}
    #             if lstm_nstep_tf_history_log and 'train_loss' in lstm_nstep_tf_history_log and 'val_loss' in lstm_nstep_tf_history_log:
    #                 # Adapt plot_loss_curves or plot directly if it expects Keras history object
    #                 # For now, let's assume plot_loss_curves can be adapted or we print metrics
    #                 print(f"Plotting loss curves for {model_name_lstm_nstep_tf} (manual history)...")
    #                 plt.figure(figsize=(10, 6))
    #                 plt.plot(lstm_nstep_tf_history_log['train_loss'], label=f'{model_name_lstm_nstep_tf} Train Loss')
    #                 plt.plot(lstm_nstep_tf_history_log['val_loss'], label=f'{model_name_lstm_nstep_tf} Validation Loss')
    #                 plt.title(f'{model_name_lstm_nstep_tf} Model Loss')
    #                 plt.ylabel('Loss (MSE)')
    #                 plt.xlabel('Epoch')
    #                 plt.legend(loc='upper right')
    #                 plt.tight_layout()
    #                 plot_filename_nstep = os.path.join(config.RESULTS_PLOTS_DIR, f'{model_name_lstm_nstep_tf}_loss_curve.png')
    #                 try:
    #                     os.makedirs(os.path.dirname(plot_filename_nstep), exist_ok=True)
    #                     plt.savefig(plot_filename_nstep)
    #                     print(f"Loss curve saved to {plot_filename_nstep}")
    #                 except Exception as e_save:
    #                     print(f"Error saving N-Step TF loss curve plot: {e_save}")
    #                 plt.show()


    #             # Rolling Forecast
    #             lstm_nstep_tf_preds_rolling_final = lstm.lstm_rolling_forecast(
    #                 train_val_series_orig, test_series_orig, lstm_nstep_tf_model, lstm_nstep_tf_scaler
    #             )
    #             if lstm_nstep_tf_preds_rolling_final is not None:
    #                 results['Rolling']['Predictions'][model_name_lstm_nstep_tf] = lstm_nstep_tf_preds_rolling_final
    #                 metrics_rolling_nstep = evaluate_performance(
    #                     f"{model_name_lstm_nstep_tf} (Rolling)", test_series_orig.values, lstm_nstep_tf_preds_rolling_final.values
    #                 )
    #                 results['Rolling']['Metrics'][model_name_lstm_nstep_tf] = metrics_rolling_nstep
    #             else:
    #                 print(f"{model_name_lstm_nstep_tf} Rolling Forecast failed.")
    #                 results['Rolling']['Metrics'][model_name_lstm_nstep_tf] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

    #             # Trajectory Forecast
    #             lstm_nstep_tf_preds_trajectory_final = lstm.lstm_trajectory_forecast(
    #                 train_val_series_orig, test_series_orig, lstm_nstep_tf_model, lstm_nstep_tf_scaler
    #             )
    #             if lstm_nstep_tf_preds_trajectory_final is not None:
    #                 results['Trajectory']['Predictions'][model_name_lstm_nstep_tf] = lstm_nstep_tf_preds_trajectory_final
    #                 metrics_trajectory_nstep = evaluate_performance(
    #                     f"{model_name_lstm_nstep_tf} (Trajectory)", test_series_orig.values, lstm_nstep_tf_preds_trajectory_final.values
    #                 )
    #                 results['Trajectory']['Metrics'][model_name_lstm_nstep_tf] = metrics_trajectory_nstep
    #             else:
    #                 print(f"{model_name_lstm_nstep_tf} Trajectory Forecast failed.")
    #                 results['Trajectory']['Metrics'][model_name_lstm_nstep_tf] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    #         else:
    #             print(f"Skipping {model_name_lstm_nstep_tf} forecasts due to training failure.")
    #             results['Rolling']['Metrics'][model_name_lstm_nstep_tf] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    #             results['Trajectory']['Metrics'][model_name_lstm_nstep_tf] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    # except Exception as e:
    #     print(f"Error during Pure LSTM N-Step TF processing: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     results['Rolling']['Metrics'][model_name_lstm_nstep_tf] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    #     results['Trajectory']['Metrics'][model_name_lstm_nstep_tf] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    # print("-" * 50)


    # --- 7. Hybrid Models ---
    print("\n[Step 7/7] Running Hybrid Models...")
    if not TF_AVAILABLE:
        print("Skipping Hybrid Models because tensorflow is not installed.")
        results['Rolling']['Metrics']['Hybrid ARIMA(1,1,1)+LSTM'] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Trajectory']['Metrics']['Hybrid ARIMA(1,1,1)+LSTM'] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Rolling']['Metrics']['Hybrid Auto ARIMA+LSTM'] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Trajectory']['Metrics']['Hybrid Auto ARIMA+LSTM'] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    else:
        # --- Select data series for Hybrid models based on config ---
        if config.HYBRID_USE_LOG_TRANSFORM:
            active_train_val_series_for_hybrid = train_val_series_log
            active_test_series_for_hybrid = test_series_log
            print_prefix_for_hybrid = "Log-Transformed"
            print("\n[Hybrid Data Prep] Using LOG-TRANSFORMED series for Hybrid models.")
        else:
            active_train_val_series_for_hybrid = train_val_series_orig
            active_test_series_for_hybrid = test_series_orig
            print_prefix_for_hybrid = "Original Scale"
            print("\n[Hybrid Data Prep] Using ORIGINAL series for Hybrid models.")

        # --- 7a. Hybrid ARIMA(1,1,1) + LSTM ---
        model_name_hybrid_111 = "Hybrid ARIMA(1,1,1)+LSTM"
        print(f"\n--- Running {model_name_hybrid_111} ({print_prefix_for_hybrid} Input) ---")
        lstm_resid_111_model, lstm_resid_111_scaler, lstm_resid_111_history = None, None, None # Initialize
        
        # ARIMA models for Hybrid must be trained on the active series
        base_arima_model_111_for_hybrid = None
        arima_model_for_hybrid_rolling_111 = None

        try:
            print(f"[Hybrid ARIMA(1,1,1)] Training base ARIMA(1,1,1) on {print_prefix_for_hybrid} data for Hybrid model...")
            base_arima_model_111_for_hybrid = arima.train_arima(active_train_val_series_for_hybrid, order=(1, 1, 1))
            
            if base_arima_model_111_for_hybrid:
                print(f"[Hybrid ARIMA(1,1,1)] Training ARIMA(1,1,1) for rolling forecast on {print_prefix_for_hybrid} data...")
                arima_model_for_hybrid_rolling_111 = arima.train_arima(active_train_val_series_for_hybrid, order=(1, 1, 1))

                if arima_model_for_hybrid_rolling_111:
                    residuals_111_for_lstm = arima.calculate_arima_residuals(active_train_val_series_for_hybrid, base_arima_model_111_for_hybrid)

                    if residuals_111_for_lstm is not None:
                        residuals_111_for_lstm = residuals_111_for_lstm.reindex(active_train_val_series_for_hybrid.index).dropna()
                        train_residuals_111_for_lstm = residuals_111_for_lstm.loc[train_series_log.index if config.HYBRID_USE_LOG_TRANSFORM else train_series_orig.index].dropna() # Index from original split
                        val_residuals_111_for_lstm = residuals_111_for_lstm.loc[val_series_log.index if config.HYBRID_USE_LOG_TRANSFORM else val_series_orig.index].dropna()   # Index from original split
                        
                        if not train_residuals_111_for_lstm.empty and not val_residuals_111_for_lstm.empty:
                            print(f"Training LSTM for {model_name_hybrid_111} ({print_prefix_for_hybrid} residuals)...")
                            lstm_resid_111_model, lstm_resid_111_scaler, lstm_resid_111_history = lstm.train_lstm(
                                train_series=train_residuals_111_for_lstm,
                                val_series=val_residuals_111_for_lstm,
                                window_size=config.LSTM_WINDOW_SIZE,
                                epochs=config.HYBRID_LSTM_EPOCHS,
                                batch_size=config.HYBRID_LSTM_BATCH_SIZE,
                                patience=config.HYBRID_LSTM_PATIENCE,
                                lstm_units_1=config.HYBRID_LSTM_UNITS_1,
                                lstm_units_2=config.HYBRID_LSTM_UNITS_2,
                                dense_units=config.HYBRID_LSTM_DENSE_UNITS,
                                standard_dropout_rate=config.LSTM_STANDARD_DROPOUT_RATE,
                                recurrent_dropout_rate=config.LSTM_RECURRENT_DROPOUT_RATE,
                                activation='relu',
                                use_recurrent_dropout=False,
                                bidirectional=False,
                                model_name=f"{model_name_hybrid_111}_Residual_{print_prefix_for_hybrid}",
                                apply_scaling=config.HYBRID_LSTM_APPLY_SCALING_TO_RESIDUALS,
                                scaler_feature_range=(-1, 1)
                           )

                            if lstm_resid_111_model:
                                # Note: trained_models stores the original log-scale ARIMA, not this hybrid-specific one.
                                # This is fine as this section is self-contained for hybrid.
                                if lstm_resid_111_history: plot_loss_curves(lstm_resid_111_history, f"{model_name_hybrid_111} Residual LSTM ({print_prefix_for_hybrid})")

                                hybrid_111_preds_rolling_output = hybrid.hybrid_rolling_forecast(
                                    active_train_val_series_for_hybrid,
                                    active_test_series_for_hybrid,
                                    arima_model_for_hybrid_rolling_111,
                                    lstm_resid_111_model,
                                    lstm_resid_111_scaler
                                )
                                if config.HYBRID_USE_LOG_TRANSFORM:
                                    hybrid_111_preds_rolling_final = np.exp(hybrid_111_preds_rolling_output) if hybrid_111_preds_rolling_output is not None else None
                                else:
                                    hybrid_111_preds_rolling_final = hybrid_111_preds_rolling_output
                                if hybrid_111_preds_rolling_final is not None:
                                    results['Rolling']['Predictions'][model_name_hybrid_111] = hybrid_111_preds_rolling_final
                                    metrics_rolling = evaluate_performance(f"{model_name_hybrid_111} (Rolling)", test_series_orig.values, hybrid_111_preds_rolling_final.values)
                                    results['Rolling']['Metrics'][model_name_hybrid_111] = metrics_rolling
                                else:
                                    print(f"{model_name_hybrid_111} Rolling Forecast failed.")
                                    results['Rolling']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

                                # Trajectory Forecast
                                hybrid_111_preds_trajectory_output = hybrid.hybrid_trajectory_forecast(
                                    active_train_val_series_for_hybrid,
                                    active_test_series_for_hybrid,
                                    base_arima_model_111_for_hybrid, # Use the one trained on active series
                                    lstm_resid_111_model,
                                    lstm_resid_111_scaler
                                )
                                if config.HYBRID_USE_LOG_TRANSFORM:
                                    hybrid_111_preds_trajectory_final = np.exp(hybrid_111_preds_trajectory_output) if hybrid_111_preds_trajectory_output is not None else None
                                else:
                                    hybrid_111_preds_trajectory_final = hybrid_111_preds_trajectory_output
                                if hybrid_111_preds_trajectory_final is not None:
                                    results['Trajectory']['Predictions'][model_name_hybrid_111] = hybrid_111_preds_trajectory_final
                                    metrics_trajectory = evaluate_performance(f"{model_name_hybrid_111} (Trajectory)", test_series_orig.values, hybrid_111_preds_trajectory_final.values)
                                    results['Trajectory']['Metrics'][model_name_hybrid_111] = metrics_trajectory
                                else:
                                    print(f"{model_name_hybrid_111} Trajectory Forecast failed.")
                                    results['Trajectory']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                                
                                # --- Plotting for Hybrid ARIMA(1,1,1)+LSTM (Adjusted for active series scale) ---
                                arima111_preds_rolling_for_plot = None
                                if arima_model_for_hybrid_rolling_111 is not None and active_test_series_for_hybrid is not None:
                                    try:
                                        print(f"[Plotting Prep] Generating ARIMA(1,1,1) rolling predictions on active_test_series_for_hybrid ({print_prefix_for_hybrid}) for component plots...")
                                        arima111_preds_rolling_for_plot = arima.arima_rolling_forecast(
                                            active_train_val_series_for_hybrid,
                                            active_test_series_for_hybrid,
                                            arima_model_for_hybrid_rolling_111
                                        )
                                    except Exception as e_arima_plot_pred:
                                        print(f"Error generating ARIMA rolling predictions for plotting for {model_name_hybrid_111}: {e_arima_plot_pred}")
                                
                                # Plot 1: ARIMA component's rolling forecast (differenced) vs. actual differenced active series
                                if arima111_preds_rolling_for_plot is not None: # Check the newly computed predictions
                                    print(f"Plotting ARIMA(1,1,1) component's rolling forecast ({print_prefix_for_hybrid}) vs differenced series for {model_name_hybrid_111}...")
                                    actual_active_series_diff = active_test_series_for_hybrid.diff().dropna()
                                    arima_preds_on_active_diff = arima111_preds_rolling_for_plot.diff().dropna()
                                    
                                    plot_predictions(
                                        actual_active_series_diff,
                                        {f'ARIMA(1,1,1) Preds Diff ({print_prefix_for_hybrid})': arima_preds_on_active_diff},
                                        title_suffix=f"{model_name_hybrid_111} - ARIMA Component ({print_prefix_for_hybrid}) vs Differenced Series",
                                        y_label=f"Differenced {print_prefix_for_hybrid} Price"
                                    )
                                else:
                                    print(f"Skipping ARIMA component plot for {model_name_hybrid_111} due to missing ARIMA predictions for plotting on active series.")

                                # Plot 2: LSTM component's rolling forecast on residuals vs. actual residuals (on active series scale)
                                actual_residuals_on_active_for_plot = None
                                # Use base_arima_model_111_for_hybrid (trained on active series) and arima111_preds_rolling_for_plot for residuals
                                if base_arima_model_111_for_hybrid and arima111_preds_rolling_for_plot is not None:
                                    actual_residuals_on_active_for_plot = (active_test_series_for_hybrid - arima111_preds_rolling_for_plot).dropna()
                                
                                if lstm_resid_111_model and actual_residuals_on_active_for_plot is not None and not actual_residuals_on_active_for_plot.empty:
                                    print(f"Generating and plotting LSTM component rolling forecast on residuals ({print_prefix_for_hybrid}) for {model_name_hybrid_111}...")
                                    
                                    # train_residuals_111_for_lstm was defined earlier and is based on the active series scale
                                    if train_residuals_111_for_lstm is not None and not train_residuals_111_for_lstm.empty:
                                        lstm_preds_rolling_residuals_on_active = lstm.lstm_rolling_forecast(
                                            train_val_series=train_residuals_111_for_lstm,
                                            test_series=actual_residuals_on_active_for_plot,
                                            model=lstm_resid_111_model,
                                            scaler=lstm_resid_111_scaler
                                        )
                                        if lstm_preds_rolling_residuals_on_active is not None:
                                            plot_predictions(
                                                actual_residuals_on_active_for_plot,
                                                {f'LSTM Residual Preds ({print_prefix_for_hybrid})': lstm_preds_rolling_residuals_on_active},
                                                title_suffix=f"{model_name_hybrid_111} - LSTM Component ({print_prefix_for_hybrid}) vs Actual Residuals",
                                                y_label=f"{print_prefix_for_hybrid} Scale Residuals"
                                            )
                                        else:
                                            print(f"LSTM component rolling forecast on residuals for {model_name_hybrid_111} failed (prediction was None).")
                                    else:
                                        print(f"Skipping LSTM component rolling forecast plot for {model_name_hybrid_111}: training residuals ('train_residuals_111_for_lstm') not available or empty.")
                                else:
                                    print(f"Skipping LSTM component rolling forecast plot for {model_name_hybrid_111} due to missing components (LSTM model or actual residuals for plotting).")
                                # --- End Plotting ---
                            else: # This else corresponds to 'if lstm_resid_111_model:'
                                print(f"Skipping {model_name_hybrid_111} forecasts due to LSTM residual training failure.")
                                results['Rolling']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                                results['Trajectory']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                        else: # This else corresponds to 'if not train_residuals_111_log.empty and not val_residuals_111_log.empty:'
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
                     # Still attempt trajectory if LSTM was trained
                     if lstm_resid_111_model and lstm_resid_111_scaler:
                          hybrid_111_preds_trajectory = hybrid.hybrid_trajectory_forecast(train_val_series, test_series, base_arima_model_111, lstm_resid_111_model, lstm_resid_111_scaler)
                          if hybrid_111_preds_trajectory is not None:
                               results['Trajectory']['Predictions'][model_name_hybrid_111] = hybrid_111_preds_trajectory
                               metrics_trajectory = evaluate_performance(f"{model_name_hybrid_111} (Trajectory)", test_series.values, hybrid_111_preds_trajectory.values)
                               results['Trajectory']['Metrics'][model_name_hybrid_111] = metrics_trajectory
                          else:
                               print(f"{model_name_hybrid_111} Trajectory Forecast failed.")
                               results['Trajectory']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                     else:
                          print(f"Skipping {model_name_hybrid_111} trajectory forecast as LSTM residual model is missing.")
                          results['Trajectory']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

            else:
                print(f"Skipping {model_name_hybrid_111} because base ARIMA(1,1,1) model training failed initially.")
                results['Rolling']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                results['Trajectory']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        except Exception as e:
            print(f"Error during Hybrid ARIMA(1,1,1) processing: {e}")
            results['Rolling']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
            results['Trajectory']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}


        # --- 7b. Hybrid Auto ARIMA + LSTM ---
        model_name_hybrid_auto = "Hybrid Auto ARIMA+LSTM"
        print(f"\n--- Running {model_name_hybrid_auto} ({print_prefix_for_hybrid} Input) ---")
        lstm_resid_auto_model, lstm_resid_auto_scaler, lstm_resid_auto_history = None, None, None # Initialize
        
        # ARIMA models for Hybrid must be trained on the active series
        base_auto_arima_model_for_hybrid = None
        auto_arima_model_for_hybrid_rolling = None

        try:
            print(f"[Hybrid Auto ARIMA] Training base Auto ARIMA on {print_prefix_for_hybrid} data for Hybrid model...")
            base_auto_arima_model_for_hybrid = arima.train_auto_arima(active_train_val_series_for_hybrid)
            
            if base_auto_arima_model_for_hybrid:
                print(f"[Hybrid Auto ARIMA] Training Auto ARIMA for rolling forecast on {print_prefix_for_hybrid} data...")
                # Need a fresh model for rolling forecast as pmdarima update() modifies in-place
                auto_arima_model_for_hybrid_rolling = arima.train_auto_arima(active_train_val_series_for_hybrid)

                if auto_arima_model_for_hybrid_rolling:
                    residuals_auto_for_lstm = arima.calculate_arima_residuals(active_train_val_series_for_hybrid, base_auto_arima_model_for_hybrid)

                    if residuals_auto_for_lstm is not None:
                        residuals_auto_for_lstm = residuals_auto_for_lstm.reindex(active_train_val_series_for_hybrid.index).dropna()
                        train_residuals_auto_for_lstm = residuals_auto_for_lstm.loc[train_series_log.index if config.HYBRID_USE_LOG_TRANSFORM else train_series_orig.index].dropna()
                        val_residuals_auto_for_lstm = residuals_auto_for_lstm.loc[val_series_log.index if config.HYBRID_USE_LOG_TRANSFORM else val_series_orig.index].dropna()

                        if not train_residuals_auto_for_lstm.empty and not val_residuals_auto_for_lstm.empty:
                            print(f"Training LSTM for {model_name_hybrid_auto} ({print_prefix_for_hybrid} residuals)...")
                            lstm_resid_auto_model, lstm_resid_auto_scaler, lstm_resid_auto_history = lstm.train_lstm(
                                train_series=train_residuals_auto_for_lstm,
                                val_series=val_residuals_auto_for_lstm,
                                window_size=config.LSTM_WINDOW_SIZE,
                                epochs=config.HYBRID_LSTM_EPOCHS,
                                batch_size=config.HYBRID_LSTM_BATCH_SIZE,
                                patience=config.HYBRID_LSTM_PATIENCE,
                                lstm_units_1=config.HYBRID_LSTM_UNITS_1,
                                lstm_units_2=config.HYBRID_LSTM_UNITS_2,
                                dense_units=config.HYBRID_LSTM_DENSE_UNITS,
                                standard_dropout_rate=config.LSTM_STANDARD_DROPOUT_RATE,
                                recurrent_dropout_rate=config.LSTM_RECURRENT_DROPOUT_RATE,
                                activation='relu',
                                use_recurrent_dropout=False,
                                bidirectional=False,
                                model_name=f"{model_name_hybrid_auto}_Residual_{print_prefix_for_hybrid}",
                                apply_scaling=config.HYBRID_LSTM_APPLY_SCALING_TO_RESIDUALS,
                                scaler_feature_range=(-1, 1)
                           )

                            if lstm_resid_auto_model:
                                # Note: trained_models stores the original log-scale Auto ARIMA.
                                if lstm_resid_auto_history: plot_loss_curves(lstm_resid_auto_history, f"{model_name_hybrid_auto} Residual LSTM ({print_prefix_for_hybrid})")

                                hybrid_auto_preds_rolling_output = hybrid.hybrid_rolling_forecast(
                                    active_train_val_series_for_hybrid,
                                    active_test_series_for_hybrid,
                                    auto_arima_model_for_hybrid_rolling,
                                    lstm_resid_auto_model,
                                    lstm_resid_auto_scaler
                                )
                                if config.HYBRID_USE_LOG_TRANSFORM:
                                    hybrid_auto_preds_rolling_final = np.exp(hybrid_auto_preds_rolling_output) if hybrid_auto_preds_rolling_output is not None else None
                                else:
                                    hybrid_auto_preds_rolling_final = hybrid_auto_preds_rolling_output
                                if hybrid_auto_preds_rolling_final is not None:
                                    results['Rolling']['Predictions'][model_name_hybrid_auto] = hybrid_auto_preds_rolling_final
                                    metrics_rolling = evaluate_performance(f"{model_name_hybrid_auto} (Rolling)", test_series_orig.values, hybrid_auto_preds_rolling_final.values)
                                    results['Rolling']['Metrics'][model_name_hybrid_auto] = metrics_rolling
                                else:
                                    print(f"{model_name_hybrid_auto} Rolling Forecast failed.")
                                    results['Rolling']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

                                # Trajectory Forecast
                                hybrid_auto_preds_trajectory_output = hybrid.hybrid_trajectory_forecast(
                                    active_train_val_series_for_hybrid,
                                    active_test_series_for_hybrid,
                                    base_auto_arima_model_for_hybrid, # Use the one trained on active series
                                    lstm_resid_auto_model,
                                    lstm_resid_auto_scaler
                                )
                                if config.HYBRID_USE_LOG_TRANSFORM:
                                    hybrid_auto_preds_trajectory_final = np.exp(hybrid_auto_preds_trajectory_output) if hybrid_auto_preds_trajectory_output is not None else None
                                else:
                                    hybrid_auto_preds_trajectory_final = hybrid_auto_preds_trajectory_output
                                if hybrid_auto_preds_trajectory_final is not None:
                                    results['Trajectory']['Predictions'][model_name_hybrid_auto] = hybrid_auto_preds_trajectory_final
                                    metrics_trajectory = evaluate_performance(f"{model_name_hybrid_auto} (Trajectory)", test_series_orig.values, hybrid_auto_preds_trajectory_final.values)
                                    results['Trajectory']['Metrics'][model_name_hybrid_auto] = metrics_trajectory
                                else:
                                    print(f"{model_name_hybrid_auto} Trajectory Forecast failed.")
                                    results['Trajectory']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                            else: # This else corresponds to 'if lstm_resid_auto_model:'
                                print(f"Skipping {model_name_hybrid_auto} forecasts due to LSTM residual training failure.")
                                results['Rolling']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                                results['Trajectory']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                        else: # This else corresponds to 'if not train_residuals_auto_log.empty and not val_residuals_auto_log.empty:'
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
                     # Still attempt trajectory if LSTM was trained
                     if lstm_resid_auto_model and lstm_resid_auto_scaler:
                          hybrid_auto_preds_trajectory = hybrid.hybrid_trajectory_forecast(train_val_series, test_series, base_auto_arima_model, lstm_resid_auto_model, lstm_resid_auto_scaler)
                          if hybrid_auto_preds_trajectory is not None:
                               results['Trajectory']['Predictions'][model_name_hybrid_auto] = hybrid_auto_preds_trajectory
                               metrics_trajectory = evaluate_performance(f"{model_name_hybrid_auto} (Trajectory)", test_series.values, hybrid_auto_preds_trajectory.values)
                               results['Trajectory']['Metrics'][model_name_hybrid_auto] = metrics_trajectory
                          else:
                               print(f"{model_name_hybrid_auto} Trajectory Forecast failed.")
                               results['Trajectory']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                     else:
                          print(f"Skipping {model_name_hybrid_auto} trajectory forecast as LSTM residual model is missing.")
                          results['Trajectory']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

            else: # This corresponds to 'if base_auto_arima_model_for_hybrid:'
                print(f"Skipping {model_name_hybrid_auto} because base Auto ARIMA model training on {print_prefix_for_hybrid} data failed.")
                results['Rolling']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                results['Trajectory']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        except Exception as e:
            print(f"Error during Hybrid Auto ARIMA processing: {e}")
            results['Rolling']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
            results['Trajectory']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

    print("-" * 50)
    print("\nHybrid Models complete.")
    # Ensure Step 8 is outside the else block of Step 7 and correctly sequenced.

    # --- 8. ICEEMDAN + Seq2Seq LSTM with Attention ---
    print("\n[Step 8/8] Running ICEEMDAN + Seq2Seq LSTM with Attention...")
    model_name_iceemdan_seq2seq = "ICEEMDAN-Seq2Seq LSTM"
    iceemdan_seq2seq_model = None
    iceemdan_history = None
    X_encoder_full, Y_decoder_input_full, Y_decoder_target_full, global_imf_scalers, global_price_scaler = None, None, None, None, None
    model_to_use_for_prediction = None # Will hold the model (loaded or trained)
    current_model_version_id = None # For naming new model if trained

    try:
        if not TF_AVAILABLE:
            print(f"Skipping {model_name_iceemdan_seq2seq} because tensorflow is not installed.")
            results['Rolling']['Metrics'][model_name_iceemdan_seq2seq] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
            results['Trajectory']['Metrics'][model_name_iceemdan_seq2seq] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        else:
            import shutil # For rmtree

            # 1. Load or Generate Processed Data (This part remains largely the same)
            full_original_series = df['adj_close'].copy()
            X_encoder_full, Y_decoder_input_full, Y_decoder_target_full, global_imf_scalers, global_price_scaler = \
                data_processing_iceemdan.generate_and_save_processed_data(
                    full_raw_price_series=full_original_series,
                    strict_train_series=train_series_orig, # Pass the strict training series for fitting scalers
                    force_reprocess=config.ICEEMDAN_DATA_FORCE_REPROCESS # Use config value
                )
            if X_encoder_full is None:
                raise ValueError("ICEEMDAN data processing failed.")

            # 2. Split Processed Data (This part remains largely the same)
            W = config.ICEEMDAN_W_WINDOW_SIZE
            train_val_end_index_orig = train_val_df.index[-1]
            split_sample_idx = -1
            for i in range(len(X_encoder_full)):
                window_end_idx_in_series = i + W - 1
                if window_end_idx_in_series < len(full_original_series):
                    if full_original_series.index[window_end_idx_in_series] <= train_val_end_index_orig:
                        split_sample_idx = i
                    else:
                        break
                else:
                    break
            if split_sample_idx == -1 or split_sample_idx + 1 >= len(X_encoder_full):
                 raise ValueError("Could not determine split point for processed ICEEMDAN data or no test samples found.")
            num_train_samples = split_sample_idx + 1
            X_encoder_train = X_encoder_full[:num_train_samples]
            Y_decoder_input_train = Y_decoder_input_full[:num_train_samples]
            Y_decoder_target_train = Y_decoder_target_full[:num_train_samples]
            X_encoder_test = X_encoder_full[num_train_samples:]
            Y_decoder_target_test = Y_decoder_target_full[num_train_samples:]
            
            print(f"[ICEEMDAN Data Prep] Splitting processed data at sample index {num_train_samples} (out of {len(X_encoder_full)}).")
            print(f"[ICEEMDAN Seq2Seq] Training data shapes: X_enc={X_encoder_train.shape}, Y_dec_in={Y_decoder_input_train.shape}, Y_dec_tgt={Y_decoder_target_train.shape}")
            print(f"[ICEEMDAN Seq2Seq] Test data shapes: X_enc={X_encoder_test.shape}, Y_dec_tgt={Y_decoder_target_test.shape}")

            # --- Model Loading/Training Decision ---
            train_model_decision = True # Default to training
            os.makedirs(config.ICEEMDAN_BASE_SAVE_DIR, exist_ok=True) # Ensure base versions directory exists

            if not config.ICEEMDAN_FORCE_RETRAIN and config.ICEEMDAN_LOAD_SAVED_MODEL:
                model_path_to_attempt_load = None
                version_to_load_str = config.ICEEMDAN_MODEL_VERSION_TO_LOAD
                
                if version_to_load_str == "latest":
                    if os.path.exists(config.ICEEMDAN_BASE_SAVE_DIR):
                        versions = sorted([
                            d for d in os.listdir(config.ICEEMDAN_BASE_SAVE_DIR)
                            if os.path.isdir(os.path.join(config.ICEEMDAN_BASE_SAVE_DIR, d))
                        ], reverse=True)
                        if versions:
                            model_path_to_attempt_load = os.path.join(config.ICEEMDAN_BASE_SAVE_DIR, versions[0])
                            print(f"Attempting to load latest model version: {versions[0]} from {model_path_to_attempt_load}")
                        else:
                            print(f"No versions found in {config.ICEEMDAN_BASE_SAVE_DIR}. Proceeding to train.")
                    else:
                        print(f"Base save directory {config.ICEEMDAN_BASE_SAVE_DIR} does not exist. Proceeding to train.")
                elif version_to_load_str == "ask" or not version_to_load_str: # "ask" or empty string triggers interactive selection
                    if os.path.exists(config.ICEEMDAN_BASE_SAVE_DIR):
                        versions = sorted([
                            d for d in os.listdir(config.ICEEMDAN_BASE_SAVE_DIR)
                            if os.path.isdir(os.path.join(config.ICEEMDAN_BASE_SAVE_DIR, d))
                        ], reverse=True)
                        if versions:
                            print("\nAvailable ICEEMDAN model versions:")
                            for i, version_id in enumerate(versions):
                                print(f"  {i+1}: {version_id}")
                            print("  0: Skip loading and retrain a new model")
                            while True:
                                try:
                                    choice = input(f"Enter number of version to load (1-{len(versions)}, or 0 to retrain): ")
                                    choice_int = int(choice)
                                    if 0 <= choice_int <= len(versions):
                                        if choice_int == 0:
                                            print("Proceeding to retrain a new model as per user choice.")
                                            # train_model_decision will remain True by default or set explicitly later
                                        else:
                                            selected_version = versions[choice_int - 1]
                                            model_path_to_attempt_load = os.path.join(config.ICEEMDAN_BASE_SAVE_DIR, selected_version)
                                            print(f"User selected version: {selected_version}. Attempting to load from {model_path_to_attempt_load}")
                                        break
                                    else:
                                        print(f"Invalid choice. Please enter a number between 0 and {len(versions)}.")
                                except ValueError:
                                    print("Invalid input. Please enter a number.")
                        else:
                            print(f"No versions found in {config.ICEEMDAN_BASE_SAVE_DIR} to select from. Proceeding to train.")
                    else:
                        print(f"Base save directory {config.ICEEMDAN_BASE_SAVE_DIR} does not exist. Proceeding to train.")
                elif version_to_load_str: # Specific version ID provided (and not "latest" or "ask")
                    model_path_to_attempt_load = os.path.join(config.ICEEMDAN_BASE_SAVE_DIR, version_to_load_str)
                    print(f"Attempting to load specified model version: {version_to_load_str} from {model_path_to_attempt_load}")
                # If version_to_load_str was empty and handled by "ask" logic, this else won't be hit for that case.
                # If it was some other non-empty, non-"latest", non-"ask" string, it's treated as specific version ID.

                if model_path_to_attempt_load and os.path.exists(model_path_to_attempt_load):
                    try:
                        print(f"Loading pre-trained full SavedModel for {model_name_iceemdan_seq2seq} from {model_path_to_attempt_load}...")
                        iceemdan_seq2seq_model = tf.keras.models.load_model(model_path_to_attempt_load, custom_objects={
                            'Seq2SeqAttentionLSTM': seq2seq_attention_lstm.Seq2SeqAttentionLSTM,
                            'Encoder': seq2seq_attention_lstm.Encoder,
                            'Decoder': seq2seq_attention_lstm.Decoder,
                            'BahdanauAttention': seq2seq_attention_lstm.BahdanauAttention
                        })
                        print("Full SavedModel loaded successfully.")
                        model_to_use_for_prediction = iceemdan_seq2seq_model
                        train_model_decision = False
                    except Exception as e:
                        print(f"Error loading full SavedModel from {model_path_to_attempt_load}: {e}. Proceeding to train.")
                        # train_model_decision remains True
                elif model_path_to_attempt_load: # Path was determined but doesn't exist
                     print(f"Specified model path {model_path_to_attempt_load} does not exist. Proceeding to train.")
                # If model_path_to_attempt_load is None (e.g. "latest" but no versions found), train_model_decision remains True
            
            elif config.ICEEMDAN_FORCE_RETRAIN:
                print("Forcing model retraining as per config.")
                # train_model_decision is already True
            else: # Not loading and not forcing retrain (e.g. LOAD_SAVED_MODEL = False)
                print("Proceeding to train as per config (LOAD_SAVED_MODEL is False or no load path determined).")
                # train_model_decision is already True

            if train_model_decision:
                current_model_version_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                print(f"Training new {model_name_iceemdan_seq2seq} model. Version ID: {current_model_version_id}")

                # Instantiate new model
                iceemdan_seq2seq_model = seq2seq_attention_lstm.Seq2SeqAttentionLSTM(
                    encoder_lstm_units=config.SEQ2SEQ_ENCODER_LSTM_UNITS,
                    decoder_lstm_units=config.SEQ2SEQ_DECODER_LSTM_UNITS,
                    output_feature_dim=1,
                    encoder_standard_dropout=config.LSTM_STANDARD_DROPOUT_RATE,
                    encoder_recurrent_dropout=config.LSTM_RECURRENT_DROPOUT_RATE,
                    decoder_standard_dropout=config.LSTM_STANDARD_DROPOUT_RATE,
                    decoder_recurrent_dropout=config.LSTM_RECURRENT_DROPOUT_RATE,
                    attention_units=config.SEQ2SEQ_ENCODER_LSTM_UNITS[-1]
                )
                optimizer = tf.keras.optimizers.Adam(learning_rate=config.SEQ2SEQ_LEARNING_RATE)
                iceemdan_seq2seq_model.compile(optimizer=optimizer, loss='mse')

                # Callbacks with versioned paths
                versioned_ckpt_dir = os.path.join(config.ICEEMDAN_BASE_SAVE_DIR, current_model_version_id, "ckpt")
                os.makedirs(versioned_ckpt_dir, exist_ok=True)
                versioned_ckpt_pattern = os.path.join(versioned_ckpt_dir, "epoch{epoch:03d}.weights.h5")
                
                model_checkpoint_weights = ModelCheckpoint(
                    filepath=versioned_ckpt_pattern,
                    monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1
                )
                early_stopping = EarlyStopping(monitor='val_loss', patience=config.SEQ2SEQ_PATIENCE, restore_best_weights=True)
                # Ensure unique log_dir for each training session, even if version_id could repeat (unlikely with timestamp)
                unique_log_suffix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                log_dir = os.path.join(config.LOGS_DIR, f"{model_name_iceemdan_seq2seq}_{current_model_version_id}_{unique_log_suffix}")
                tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
                callbacks_list = [early_stopping, model_checkpoint_weights, tensorboard_callback]

                iceemdan_history = iceemdan_seq2seq_model.fit(
                    x=[X_encoder_train, Y_decoder_input_train], y=Y_decoder_target_train,
                    epochs=config.SEQ2SEQ_EPOCHS, batch_size=config.SEQ2SEQ_BATCH_SIZE,
                    validation_split=0.2, callbacks=callbacks_list, verbose=1
                )
                print("Training complete.")
                
                save_choice = input(f"ICEEMDAN model (Version ID: {current_model_version_id}) has been trained. Save this model? (yes/no): ").strip().lower()
                
                final_model_save_path = os.path.join(config.ICEEMDAN_BASE_SAVE_DIR, current_model_version_id)

                if save_choice in ['yes', 'y']:
                    # Ensure parent directory for the version exists (os.makedirs for ckpt might have done this)
                    os.makedirs(config.ICEEMDAN_BASE_SAVE_DIR, exist_ok=True)
                    # model.save will create the final_model_save_path directory itself.
                    # If it somehow exists (e.g. from a previous interrupted save of the exact same version_id), clean it.
                    if os.path.exists(final_model_save_path):
                        print(f"Warning: Target save directory {final_model_save_path} already exists. Removing before saving.")
                        shutil.rmtree(final_model_save_path) # Remove directory for a clean save
                    
                    print(f"Saving full model to {final_model_save_path}...")
                    iceemdan_seq2seq_model.save(final_model_save_path)
                    print(f"Full model saved to {final_model_save_path}")
                    model_to_use_for_prediction = iceemdan_seq2seq_model
                else:
                    print(f"Model (Version ID: {current_model_version_id}) was NOT saved as per user choice.")
                    model_to_use_for_prediction = iceemdan_seq2seq_model # Still use it for current session's prediction
                    # Clean up the entire version directory for this unsaved version (includes ckpt)
                    if os.path.exists(final_model_save_path): # This is the version_id directory
                        print(f"Cleaning up directory for unsaved version: {final_model_save_path}")
                        try:
                            shutil.rmtree(final_model_save_path)
                        except OSError as e_rm:
                            print(f"Error cleaning up directory {final_model_save_path}: {e_rm}")
                    elif os.path.exists(versioned_ckpt_dir): # If only ckpt_dir was created
                         # versioned_ckpt_dir is .../version_id/ckpt. We need to remove .../version_id
                         parent_version_dir = os.path.dirname(versioned_ckpt_dir)
                         if os.path.basename(parent_version_dir) == current_model_version_id: # Sanity check
                            print(f"Cleaning up checkpoint parent directory for unsaved version: {parent_version_dir}")
                            try:
                                shutil.rmtree(parent_version_dir)
                            except OSError as e_rm:
                                print(f"Error cleaning up directory {parent_version_dir}: {e_rm}")
                
                if iceemdan_history: # Plot loss regardless of saving the model itself
                    plot_loss_curves(iceemdan_history, f"{model_name_iceemdan_seq2seq} (Version: {current_model_version_id})")
            
            # Ensure iceemdan_seq2seq_model variable (used by prediction logic below) points to the correct model
            if model_to_use_for_prediction is not None:
                iceemdan_seq2seq_model = model_to_use_for_prediction
            elif not train_model_decision: # Failed to load and did not retrain
                 print(f"Critical: Failed to load model and retraining was not initiated. ICEEMDAN predictions will be skipped.")
                 # Set metrics to NaN or handle appropriately
                 results['Rolling']['Metrics'][model_name_iceemdan_seq2seq] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                 results['Trajectory']['Metrics'][model_name_iceemdan_seq2seq] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                 # Ensure prediction part is skipped if model is None
                 iceemdan_seq2seq_model = None # Explicitly set to None

            # 7. Prediction on Test Set (Trajectory Forecast)
            print(f"Generating trajectory forecast for {model_name_iceemdan_seq2seq}...")
            test_predictions_scaled_list = []
            num_test_samples = X_encoder_test.shape[0]

            for i in range(num_test_samples):
                # Get the single encoder input sample
                current_encoder_input = X_encoder_test[i:i+1] # Shape (1, encoder_timesteps, K+1)

                # Get the true scaled price at time T for the first decoder input
                # This corresponds to the end of the window for the *previous* training sample
                # Window i in the original full set ends at index i + W - 1
                # The first test sample has index `num_train_samples` in the full set.
                # Its window ends at `num_train_samples + W - 1`.
                # So, y_T for the i-th test sample (original index num_train_samples + i)
                # corresponds to the price at original index (num_train_samples + i) + W - 1
                original_series_idx_for_yT = (num_train_samples + i) + W - 1
                if original_series_idx_for_yT < len(full_original_series):
                    true_y_T_original = full_original_series.iloc[original_series_idx_for_yT]
                    true_y_T_scaled = global_price_scaler.transform(np.array([[true_y_T_original]]))[0,0]
                else:
                    # Handle edge case where index might be out of bounds (shouldn't happen ideally)
                    print(f"Warning: Could not find y_T for test sample {i}. Using last available price.")
                    true_y_T_scaled = scaled_raw_prices[-1] # Fallback

                # Perform H-step prediction using the model's predict_sequence method
                predicted_scaled_sequence = iceemdan_seq2seq_model.predict_sequence(
                    current_encoder_input,
                    tf.constant([[true_y_T_scaled]], dtype=tf.float32), # Ensure correct shape and type
                    config.ICEEMDAN_H_FORECAST_PERIOD
                ) # Output shape (1, H, 1)

                test_predictions_scaled_list.append(predicted_scaled_sequence.numpy().flatten()) # Flatten to (H,)

            # The loop for `num_test_samples` was for an alternative way of multi-step prediction,
            # not for a single continuous trajectory or rolling forecast.
            # We will implement rolling forecast separately below.

            # Trajectory Forecast (single continuous prediction for the whole test period)
            print(f"Generating single trajectory forecast for {model_name_iceemdan_seq2seq} for the entire test period...")
            if X_encoder_train.shape[0] > 0: # Ensure there is training data to get the last input
                last_train_encoder_input = X_encoder_train[-1:]
                last_train_yT_index = (num_train_samples - 1) + W - 1 # Index in full_original_series
                if last_train_yT_index < len(full_original_series):
                    last_train_yT_original = full_original_series.iloc[last_train_yT_index]
                    last_train_yT_scaled = global_price_scaler.transform(np.array([[last_train_yT_original]]))[0,0]

                    full_trajectory_scaled = iceemdan_seq2seq_model.predict_sequence(
                        last_train_encoder_input,
                        tf.constant([[last_train_yT_scaled]], dtype=tf.float32),
                        len(test_series_orig)
                    )
                    full_trajectory_final_values = global_price_scaler.inverse_transform(full_trajectory_scaled[0].numpy())
                    iceemdan_preds_trajectory_final = pd.Series(full_trajectory_final_values.flatten(), index=test_series_orig.index)
                    
                    results['Trajectory']['Predictions'][model_name_iceemdan_seq2seq] = iceemdan_preds_trajectory_final
                    metrics_trajectory = evaluate_performance(f"{model_name_iceemdan_seq2seq} (Trajectory)", test_series_orig.values, iceemdan_preds_trajectory_final.values)
                    results['Trajectory']['Metrics'][model_name_iceemdan_seq2seq] = metrics_trajectory
                else:
                    print(f"Error: Could not get last y_T for trajectory forecast. Index out of bounds.")
                    results['Trajectory']['Metrics'][model_name_iceemdan_seq2seq] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
            else:
                print("Error: No training data available for trajectory forecast base.")
                results['Trajectory']['Metrics'][model_name_iceemdan_seq2seq] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

            # 8. Rolling Forecast for ICEEMDAN-Seq2Seq
            print(f"\nGenerating rolling forecast for {model_name_iceemdan_seq2seq}...")
            iceemdan_rolling_predictions_scaled = []
            
            # The number of rolling predictions will be min(len(X_encoder_test), len(test_series_orig))
            # because each X_encoder_test sample gives the first step of an H-step forecast.
            num_rolling_preds = min(len(X_encoder_test), len(test_series_orig))

            if num_rolling_preds > 0:
                for i in range(num_rolling_preds):
                    current_encoder_input_rolling = X_encoder_test[i:i+1] # Shape (1, encoder_timesteps, K+1)
                    
                    # Determine y_T for this rolling step's decoder:
                    # It's the price at the end of the window that current_encoder_input_rolling represents.
                    # The i-th sample in X_encoder_test corresponds to the (num_train_samples + i)-th sample in X_encoder_full.
                    # The window for this sample ends at original series index: (num_train_samples + i) + W - 1
                    original_series_idx_for_yT_rolling = (num_train_samples + i) + W - 1
                    
                    if original_series_idx_for_yT_rolling < len(full_original_series):
                        true_y_T_original_rolling = full_original_series.iloc[original_series_idx_for_yT_rolling]
                        true_y_T_scaled_rolling = global_price_scaler.transform(np.array([[true_y_T_original_rolling]]))[0,0]

                        # Predict H steps using the model
                        predicted_h_steps_scaled = iceemdan_seq2seq_model.predict_sequence(
                            current_encoder_input_rolling,
                            tf.constant([[true_y_T_scaled_rolling]], dtype=tf.float32),
                            config.ICEEMDAN_H_FORECAST_PERIOD # Predict H steps
                        ) # Output shape (1, H, 1)
                        
                        # Take only the first step of the H-step prediction for rolling forecast
                        first_step_prediction_scaled = predicted_h_steps_scaled.numpy()[0, 0, 0]
                        iceemdan_rolling_predictions_scaled.append(first_step_prediction_scaled)
                    else:
                        print(f"Warning: Skipping rolling forecast for test step {i} due to y_T index out of bounds.")
                        iceemdan_rolling_predictions_scaled.append(np.nan)
                
                # Inverse transform the collected rolling predictions
                if iceemdan_rolling_predictions_scaled:
                    iceemdan_rolling_predictions_scaled_np = np.array(iceemdan_rolling_predictions_scaled).reshape(-1, 1)
                    iceemdan_rolling_predictions_final_values = global_price_scaler.inverse_transform(iceemdan_rolling_predictions_scaled_np)
                    
                    # Create a pandas Series with the correct index from test_series_orig
                    # The predictions correspond to the first num_rolling_preds of test_series_orig
                    iceemdan_preds_rolling_final = pd.Series(
                        iceemdan_rolling_predictions_final_values.flatten(),
                        index=test_series_orig.index[:num_rolling_preds]
                    ).reindex(test_series_orig.index) # Reindex to fill with NaNs if shorter

                    results['Rolling']['Predictions'][model_name_iceemdan_seq2seq] = iceemdan_preds_rolling_final
                    metrics_rolling = evaluate_performance(
                        f"{model_name_iceemdan_seq2seq} (Rolling)",
                        test_series_orig.values, # True values
                        iceemdan_preds_rolling_final.fillna(0).values # Predictions, fill NaNs for evaluation if any
                    )
                    results['Rolling']['Metrics'][model_name_iceemdan_seq2seq] = metrics_rolling
                else:
                    print(f"{model_name_iceemdan_seq2seq} Rolling Forecast produced no predictions.")
                    results['Rolling']['Metrics'][model_name_iceemdan_seq2seq] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                    results['Rolling']['Predictions'][model_name_iceemdan_seq2seq] = None
            else:
                print(f"Skipping {model_name_iceemdan_seq2seq} Rolling Forecast as X_encoder_test is empty or test_series_orig is too short.")
                results['Rolling']['Metrics'][model_name_iceemdan_seq2seq] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                results['Rolling']['Predictions'][model_name_iceemdan_seq2seq] = None

    except Exception as e:
        # Indented block for the except statement
        print(f"Error during {model_name_iceemdan_seq2seq} processing: {e}")
        import traceback
        traceback.print_exc()
        results['Rolling']['Metrics'][model_name_iceemdan_seq2seq] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Trajectory']['Metrics'][model_name_iceemdan_seq2seq] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

        # === ICEEMDAN ACC DEBUG BLOCK (用完後整段刪除) ===
    import numpy as np

    # 1. 拿出真實值與預測值
    y_true = test_series_orig.values
    y_pred = iceemdan_preds_trajectory_final.values

    # 2. 去除 NaN
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_v = y_true[mask]
    y_pred_v = y_pred[mask]

    # 3. 計算「漲跌」符號
    true_diff = np.diff(y_true_v)
    pred_diff = np.diff(y_pred_v)

    # 4. 印出前 10 筆對照
    print("【ICEEMDAN 真實前10筆】", y_true_v[:10])
    print("【ICEEMDAN 預測前10筆】", y_pred_v[:10])
    print("【真實漲跌符號】", np.sign(true_diff)[:10])
    print("【預測漲跌符號】", np.sign(pred_diff)[:10])

# 5. 計算並印出方向準確率
    acc = np.mean(np.sign(true_diff) == np.sign(pred_diff)) * 100
    print(f"【DEBUG】ICEEMDAN 方向準確率 = {acc:.2f}%")
# === END DEBUG BLOCK ===


    print("-" * 50) # This was the end of Step 8's try...except

    # --- Final Results Aggregation and Visualization ---
    # This section should come AFTER all model steps.
    print("\n--- Aggregating and Visualizing Results ---")

    # Plot Rolling Forecasts - use original scale test series and final predictions
    plot_predictions(test_series_orig, results['Rolling']['Predictions'], title_suffix="Rolling Forecast")

    # Plot Trajectory Forecasts - use original scale test series and final predictions
    plot_predictions(test_series_orig, results['Trajectory']['Predictions'], title_suffix="Trajectory Forecast")

    # Display Metrics Tables (metrics were calculated on original scale)
    print("\n--- Rolling Forecast Metrics ---")
    try:
        rolling_metrics_df = pd.DataFrame(results['Rolling']['Metrics']).T.sort_index()
        print(rolling_metrics_df.to_string(float_format="%.4f"))
    except Exception as e:
        print(f"Could not display rolling metrics: {e}")


    print("\n--- Trajectory Forecast Metrics ---")
    try:
        trajectory_metrics_df = pd.DataFrame(results['Trajectory']['Metrics']).T.sort_index()
        print(trajectory_metrics_df.to_string(float_format="%.4f"))
    except Exception as e:
        print(f"Could not display trajectory metrics: {e}")


    print("\n--- NVDA Prediction Analysis End ---")
    # Optional: Control plt.show() behavior if running non-interactively
    # plt.show() # Already commented out in visualization.py