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
try:
    # Use relative imports within the package
    from . import config
    from .data_processing import load_and_preprocess_data, split_data
    from .evaluation import evaluate_performance
    from .visualization import plot_loss_curves, plot_predictions, plot_full_history # Import new function
    from .models import naive, arima, lstm, hybrid # Import model modules
    from .models.lstm import TF_AVAILABLE # Check TF availability
except ImportError:
    # Fallback for direct execution (e.g., running main.py directly)
    import config
    from data_processing import load_and_preprocess_data, split_data
    from evaluation import evaluate_performance
    from visualization import plot_loss_curves, plot_predictions
    import models.naive as naive
    import models.arima as arima
    import models.lstm as lstm
    import models.hybrid as hybrid
    from models.lstm import TF_AVAILABLE

# --- Configuration ---
warnings.filterwarnings("ignore") # Ignore harmless warnings
# Plot style is applied in visualization.py

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
                dropout_rate=config.LSTM_DROPOUT_RATE,
                activation='tanh', # Explicitly set for Pure LSTM
                use_recurrent_dropout=True, # Explicitly set for Pure LSTM
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


    # --- 7. Hybrid Models ---
    print("\n[Step 7/7] Running Hybrid Models...")
    if not TF_AVAILABLE:
        print("Skipping Hybrid Models because tensorflow is not installed.")
        results['Rolling']['Metrics']['Hybrid ARIMA(1,1,1)+LSTM'] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Trajectory']['Metrics']['Hybrid ARIMA(1,1,1)+LSTM'] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Rolling']['Metrics']['Hybrid Auto ARIMA+LSTM'] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        results['Trajectory']['Metrics']['Hybrid Auto ARIMA+LSTM'] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
    else:
        # --- 7a. Hybrid ARIMA(1,1,1) + LSTM ---
        model_name_hybrid_111 = "Hybrid ARIMA(1,1,1)+LSTM"
        print(f"\n--- Running {model_name_hybrid_111} ---")
        lstm_resid_111_model, lstm_resid_111_scaler, lstm_resid_111_history = None, None, None # Initialize
        arima_model_for_hybrid_rolling_111 = None # Initialize

        try:
            if model_name_arima111 in trained_models:
                base_arima_model_111 = trained_models[model_name_arima111]
                # Need a fresh model instance for rolling forecast if using statsmodels extend
                # Retrain ARIMA(1,1,1) specifically for rolling hybrid on log-transformed data
                arima_model_for_hybrid_rolling_111 = arima.train_arima(train_val_series_log, order=(1, 1, 1))

                if arima_model_for_hybrid_rolling_111: # Check if retraining for rolling worked
                    # Residuals are calculated on log-transformed data
                    residuals_111_log = arima.calculate_arima_residuals(train_val_series_log, base_arima_model_111) # base_arima_model_111 was trained on log data

                    if residuals_111_log is not None:
                        # Align residuals index with train_val_series_log for splitting
                        residuals_111_log = residuals_111_log.reindex(train_val_series_log.index).dropna()
                        # Split residuals based on train/val split indices (derived from log series)
                        train_residuals_111_log = residuals_111_log.loc[train_series_log.index].dropna()
                        val_residuals_111_log = residuals_111_log.loc[val_series_log.index].dropna()

                        if not train_residuals_111_log.empty and not val_residuals_111_log.empty:
                            print(f"Training LSTM for {model_name_hybrid_111} log-residuals...")
                            # Train LSTM on log-residuals using Hybrid LSTM parameters
                            lstm_resid_111_model, lstm_resid_111_scaler, lstm_resid_111_history = lstm.train_lstm(
                                train_series=train_residuals_111_log, # Pass log-residuals
                                val_series=val_residuals_111_log,
                                # Pass Hybrid LSTM specific parameters from config
                                window_size=config.LSTM_WINDOW_SIZE,
                                epochs=config.HYBRID_LSTM_EPOCHS,
                                batch_size=config.HYBRID_LSTM_BATCH_SIZE,
                                patience=config.HYBRID_LSTM_PATIENCE,
                                lstm_units_1=config.HYBRID_LSTM_UNITS_1,
                                lstm_units_2=config.HYBRID_LSTM_UNITS_2,
                                dense_units=config.HYBRID_LSTM_DENSE_UNITS,
                                dropout_rate=config.LSTM_DROPOUT_RATE,
                                activation='relu', # Explicitly set for Hybrid LSTM
                                use_recurrent_dropout=False, # Explicitly set for Hybrid LSTM
                                bidirectional=False, # Ensure Hybrid uses unidirectional
                                model_name=f"{model_name_hybrid_111}_Residual" # Added model name
                            )

                            if lstm_resid_111_model and lstm_resid_111_scaler: # Scaler is for log-residuals
                                trained_models[model_name_hybrid_111] = {'arima': base_arima_model_111, 'lstm': lstm_resid_111_model, 'scaler': lstm_resid_111_scaler}
                                if lstm_resid_111_history: plot_loss_curves(lstm_resid_111_history, f"{model_name_hybrid_111} Residual LSTM") # Loss is on log-residuals

                                # Rolling Forecast - Hybrid still operates on log scale, needs np.exp()
                                hybrid_111_preds_rolling_log = hybrid.hybrid_rolling_forecast(train_val_series_log, test_series_log, arima_model_for_hybrid_rolling_111, lstm_resid_111_model, lstm_resid_111_scaler)
                                hybrid_111_preds_rolling_final = np.exp(hybrid_111_preds_rolling_log) if hybrid_111_preds_rolling_log is not None else None
                                if hybrid_111_preds_rolling_final is not None:
                                    results['Rolling']['Predictions'][model_name_hybrid_111] = hybrid_111_preds_rolling_final
                                    metrics_rolling = evaluate_performance(f"{model_name_hybrid_111} (Rolling)", test_series_orig.values, hybrid_111_preds_rolling_final.values)
                                    results['Rolling']['Metrics'][model_name_hybrid_111] = metrics_rolling
                                else:
                                    print(f"{model_name_hybrid_111} Rolling Forecast failed.")
                                    results['Rolling']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

                                # Trajectory Forecast - Hybrid still operates on log scale, needs np.exp()
                                hybrid_111_preds_trajectory_log = hybrid.hybrid_trajectory_forecast(train_val_series_log, test_series_log, base_arima_model_111, lstm_resid_111_model, lstm_resid_111_scaler)
                                hybrid_111_preds_trajectory_final = np.exp(hybrid_111_preds_trajectory_log) if hybrid_111_preds_trajectory_log is not None else None
                                if hybrid_111_preds_trajectory_final is not None:
                                    results['Trajectory']['Predictions'][model_name_hybrid_111] = hybrid_111_preds_trajectory_final
                                    metrics_trajectory = evaluate_performance(f"{model_name_hybrid_111} (Trajectory)", test_series_orig.values, hybrid_111_preds_trajectory_final.values)
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
        print(f"\n--- Running {model_name_hybrid_auto} ---")
        lstm_resid_auto_model, lstm_resid_auto_scaler, lstm_resid_auto_history = None, None, None # Initialize
        auto_arima_model_for_hybrid_rolling = None # Initialize

        try:
            if model_name_auto_arima in trained_models:
                base_auto_arima_model = trained_models[model_name_auto_arima]
                # Need a fresh model for rolling forecast as pmdarima update() modifies in-place
                auto_arima_model_for_hybrid_rolling = arima.train_auto_arima(train_val_series_log) # Retrain on log-data

                if auto_arima_model_for_hybrid_rolling: # Check if retraining for rolling worked
                    residuals_auto_log = arima.calculate_arima_residuals(train_val_series_log, base_auto_arima_model) # base_auto_arima_model trained on log

                    if residuals_auto_log is not None:
                        residuals_auto_log = residuals_auto_log.reindex(train_val_series_log.index).dropna()
                        train_residuals_auto_log = residuals_auto_log.loc[train_series_log.index].dropna()
                        val_residuals_auto_log = residuals_auto_log.loc[val_series_log.index].dropna()

                        if not train_residuals_auto_log.empty and not val_residuals_auto_log.empty:
                            print(f"Training LSTM for {model_name_hybrid_auto} log-residuals...")
                            # Train LSTM on log-residuals using Hybrid LSTM parameters
                            lstm_resid_auto_model, lstm_resid_auto_scaler, lstm_resid_auto_history = lstm.train_lstm(
                                train_series=train_residuals_auto_log, # Pass log-residuals
                                val_series=val_residuals_auto_log,
                                # Pass Hybrid LSTM specific parameters from config
                                window_size=config.LSTM_WINDOW_SIZE,
                                epochs=config.HYBRID_LSTM_EPOCHS,
                                batch_size=config.HYBRID_LSTM_BATCH_SIZE,
                                patience=config.HYBRID_LSTM_PATIENCE,
                                lstm_units_1=config.HYBRID_LSTM_UNITS_1,
                                lstm_units_2=config.HYBRID_LSTM_UNITS_2,
                                dense_units=config.HYBRID_LSTM_DENSE_UNITS,
                                dropout_rate=config.LSTM_DROPOUT_RATE,
                                activation='relu', # Explicitly set for Hybrid LSTM
                                use_recurrent_dropout=False, # Explicitly set for Hybrid LSTM
                                bidirectional=False, # Ensure Hybrid uses unidirectional
                                model_name=f"{model_name_hybrid_auto}_Residual" # Added model name
                            )

                            if lstm_resid_auto_model and lstm_resid_auto_scaler: # Scaler for log-residuals
                                trained_models[model_name_hybrid_auto] = {'arima': base_auto_arima_model, 'lstm': lstm_resid_auto_model, 'scaler': lstm_resid_auto_scaler}
                                if lstm_resid_auto_history: plot_loss_curves(lstm_resid_auto_history, f"{model_name_hybrid_auto} Residual LSTM") # Loss on log-residuals

                                # Rolling Forecast - Hybrid still operates on log scale, needs np.exp()
                                hybrid_auto_preds_rolling_log = hybrid.hybrid_rolling_forecast(train_val_series_log, test_series_log, auto_arima_model_for_hybrid_rolling, lstm_resid_auto_model, lstm_resid_auto_scaler)
                                hybrid_auto_preds_rolling_final = np.exp(hybrid_auto_preds_rolling_log) if hybrid_auto_preds_rolling_log is not None else None
                                if hybrid_auto_preds_rolling_final is not None:
                                    results['Rolling']['Predictions'][model_name_hybrid_auto] = hybrid_auto_preds_rolling_final
                                    metrics_rolling = evaluate_performance(f"{model_name_hybrid_auto} (Rolling)", test_series_orig.values, hybrid_auto_preds_rolling_final.values)
                                    results['Rolling']['Metrics'][model_name_hybrid_auto] = metrics_rolling
                                else:
                                    print(f"{model_name_hybrid_auto} Rolling Forecast failed.")
                                    results['Rolling']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

                                # Trajectory Forecast - Hybrid still operates on log scale, needs np.exp()
                                hybrid_auto_preds_trajectory_log = hybrid.hybrid_trajectory_forecast(train_val_series_log, test_series_log, base_auto_arima_model, lstm_resid_auto_model, lstm_resid_auto_scaler)
                                hybrid_auto_preds_trajectory_final = np.exp(hybrid_auto_preds_trajectory_log) if hybrid_auto_preds_trajectory_log is not None else None
                                if hybrid_auto_preds_trajectory_final is not None:
                                    results['Trajectory']['Predictions'][model_name_hybrid_auto] = hybrid_auto_preds_trajectory_final
                                    metrics_trajectory = evaluate_performance(f"{model_name_hybrid_auto} (Trajectory)", test_series_orig.values, hybrid_auto_preds_trajectory_final.values)
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

            else:
                print(f"Skipping {model_name_hybrid_auto} because base Auto ARIMA model training failed initially.")
                results['Rolling']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
                results['Trajectory']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
        except Exception as e:
            print(f"Error during Hybrid Auto ARIMA processing: {e}")
            results['Rolling']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
            results['Trajectory']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

    print("-" * 50)
    print("\nHybrid Models complete.")

    # --- Final Results Aggregation and Visualization ---
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