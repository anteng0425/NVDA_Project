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
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

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
    trained_models = {} # Store trained models for potential reuse (e.g., in hybrid)

    print("\nData loading and splitting complete. Ready for model training and prediction.")
    print("-" * 50)

    # --- 3. Naive Forecast ---
    print("\n[Step 3/7] Running Naive Forecast...")
    model_name_naive = "Naive Forecast"
    try:
        naive_preds_rolling = naive.naive_forecast_rolling(test_series)
        results['Rolling']['Predictions'][model_name_naive] = naive_preds_rolling
        metrics_rolling_naive = evaluate_performance(f"{model_name_naive} (Rolling)", test_series.values, naive_preds_rolling.values if naive_preds_rolling is not None else None)
        results['Rolling']['Metrics'][model_name_naive] = metrics_rolling_naive

        naive_preds_trajectory = naive.naive_forecast_trajectory(train_val_series, test_series)
        results['Trajectory']['Predictions'][model_name_naive] = naive_preds_trajectory
        metrics_trajectory_naive = evaluate_performance(f"{model_name_naive} (Trajectory)", test_series.values, naive_preds_trajectory.values if naive_preds_trajectory is not None else None)
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
        # Ensure train_val_series has frequency for statsmodels (handled in data_processing)
        arima111_model = arima.train_arima(train_val_series, order=(1, 1, 1))

        if arima111_model:
            trained_models[model_name_arima111] = arima111_model
            # Rolling Forecast
            arima111_preds_rolling = arima.arima_rolling_forecast(train_val_series, test_series, arima111_model)
            if arima111_preds_rolling is not None:
                results['Rolling']['Predictions'][model_name_arima111] = arima111_preds_rolling
                metrics_rolling = evaluate_performance(f"{model_name_arima111} (Rolling)", test_series.values, arima111_preds_rolling.values)
                results['Rolling']['Metrics'][model_name_arima111] = metrics_rolling
            else:
                print(f"{model_name_arima111} Rolling Forecast failed.")
                results['Rolling']['Metrics'][model_name_arima111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

            # Trajectory Forecast
            arima111_preds_trajectory = arima.arima_trajectory_forecast(train_val_series, test_series, arima111_model)
            if arima111_preds_trajectory is not None:
                results['Trajectory']['Predictions'][model_name_arima111] = arima111_preds_trajectory
                metrics_trajectory = evaluate_performance(f"{model_name_arima111} (Trajectory)", test_series.values, arima111_preds_trajectory.values)
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
        # Uses parameters from config module
        auto_arima_model = arima.train_auto_arima(train_val_series)

        if auto_arima_model:
            trained_models[model_name_auto_arima] = auto_arima_model
            # Need a fresh model for rolling forecast as pmdarima update() modifies in-place
            auto_arima_model_for_rolling = arima.train_auto_arima(train_val_series) # Retrain

            if auto_arima_model_for_rolling:
                auto_arima_preds_rolling = arima.arima_rolling_forecast(train_val_series, test_series, auto_arima_model_for_rolling)
                if auto_arima_preds_rolling is not None:
                    results['Rolling']['Predictions'][model_name_auto_arima] = auto_arima_preds_rolling
                    metrics_rolling = evaluate_performance(f"{model_name_auto_arima} (Rolling)", test_series.values, auto_arima_preds_rolling.values)
                    results['Rolling']['Metrics'][model_name_auto_arima] = metrics_rolling
                else:
                    print(f"{model_name_auto_arima} Rolling Forecast failed.")
                    results['Rolling']['Metrics'][model_name_auto_arima] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}
            else:
                 print(f"Failed to get a fresh Auto ARIMA model for rolling forecast.")
                 results['Rolling']['Metrics'][model_name_auto_arima] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

            # Trajectory Forecast (use original trained model)
            auto_arima_preds_trajectory = arima.arima_trajectory_forecast(train_val_series, test_series, auto_arima_model)
            if auto_arima_preds_trajectory is not None:
                results['Trajectory']['Predictions'][model_name_auto_arima] = auto_arima_preds_trajectory
                metrics_trajectory = evaluate_performance(f"{model_name_auto_arima} (Trajectory)", test_series.values, auto_arima_preds_trajectory.values)
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
            # Uses hyperparameters from config module
            lstm_model, lstm_scaler, lstm_history = lstm.train_lstm(
                train_series, val_series # Pass prices directly
            )

            if lstm_model and lstm_scaler: # Check if training was successful
                trained_models[model_name_lstm] = {'model': lstm_model, 'scaler': lstm_scaler}
                if lstm_history: plot_loss_curves(lstm_history, model_name_lstm)

                # Rolling Forecast
                lstm_preds_rolling = lstm.lstm_rolling_forecast(train_val_series, test_series, lstm_model, lstm_scaler)
                if lstm_preds_rolling is not None:
                    results['Rolling']['Predictions'][model_name_lstm] = lstm_preds_rolling
                    metrics_rolling = evaluate_performance(f"{model_name_lstm} (Rolling)", test_series.values, lstm_preds_rolling.values)
                    results['Rolling']['Metrics'][model_name_lstm] = metrics_rolling
                else:
                    print(f"{model_name_lstm} Rolling Forecast failed.")
                    results['Rolling']['Metrics'][model_name_lstm] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

                # Trajectory Forecast
                lstm_preds_trajectory = lstm.lstm_trajectory_forecast(train_val_series, test_series, lstm_model, lstm_scaler)
                if lstm_preds_trajectory is not None:
                    results['Trajectory']['Predictions'][model_name_lstm] = lstm_preds_trajectory
                    metrics_trajectory = evaluate_performance(f"{model_name_lstm} (Trajectory)", test_series.values, lstm_preds_trajectory.values)
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
                # Retrain ARIMA(1,1,1) specifically for rolling hybrid
                arima_model_for_hybrid_rolling_111 = arima.train_arima(train_val_series, order=(1, 1, 1))

                if arima_model_for_hybrid_rolling_111: # Check if retraining for rolling worked
                    residuals_111 = arima.calculate_arima_residuals(train_val_series, base_arima_model_111)

                    if residuals_111 is not None:
                        # Align residuals index with train_val_series for splitting
                        residuals_111 = residuals_111.reindex(train_val_series.index).dropna()
                        # Split residuals based on train/val split indices
                        train_residuals_111 = residuals_111.loc[train_series.index].dropna()
                        val_residuals_111 = residuals_111.loc[val_series.index].dropna()

                        if not train_residuals_111.empty and not val_residuals_111.empty:
                            print(f"Training LSTM for {model_name_hybrid_111} residuals...")
                            # Train LSTM on residuals
                            lstm_resid_111_model, lstm_resid_111_scaler, lstm_resid_111_history = lstm.train_lstm(
                                train_residuals_111, val_residuals_111 # Pass residuals
                            )

                            if lstm_resid_111_model and lstm_resid_111_scaler:
                                trained_models[model_name_hybrid_111] = {'arima': base_arima_model_111, 'lstm': lstm_resid_111_model, 'scaler': lstm_resid_111_scaler}
                                if lstm_resid_111_history: plot_loss_curves(lstm_resid_111_history, f"{model_name_hybrid_111} Residual LSTM")

                                # Rolling Forecast - Use the fresh ARIMA model instance
                                hybrid_111_preds_rolling = hybrid.hybrid_rolling_forecast(train_val_series, test_series, arima_model_for_hybrid_rolling_111, lstm_resid_111_model, lstm_resid_111_scaler)
                                if hybrid_111_preds_rolling is not None:
                                    results['Rolling']['Predictions'][model_name_hybrid_111] = hybrid_111_preds_rolling
                                    metrics_rolling = evaluate_performance(f"{model_name_hybrid_111} (Rolling)", test_series.values, hybrid_111_preds_rolling.values)
                                    results['Rolling']['Metrics'][model_name_hybrid_111] = metrics_rolling
                                else:
                                    print(f"{model_name_hybrid_111} Rolling Forecast failed.")
                                    results['Rolling']['Metrics'][model_name_hybrid_111] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

                                # Trajectory Forecast - Use the original base ARIMA model
                                hybrid_111_preds_trajectory = hybrid.hybrid_trajectory_forecast(train_val_series, test_series, base_arima_model_111, lstm_resid_111_model, lstm_resid_111_scaler)
                                if hybrid_111_preds_trajectory is not None:
                                    results['Trajectory']['Predictions'][model_name_hybrid_111] = hybrid_111_preds_trajectory
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
                auto_arima_model_for_hybrid_rolling = arima.train_auto_arima(train_val_series) # Retrain

                if auto_arima_model_for_hybrid_rolling: # Check if retraining for rolling worked
                    residuals_auto = arima.calculate_arima_residuals(train_val_series, base_auto_arima_model)

                    if residuals_auto is not None:
                        residuals_auto = residuals_auto.reindex(train_val_series.index).dropna()
                        train_residuals_auto = residuals_auto.loc[train_series.index].dropna()
                        val_residuals_auto = residuals_auto.loc[val_series.index].dropna()

                        if not train_residuals_auto.empty and not val_residuals_auto.empty:
                            print(f"Training LSTM for {model_name_hybrid_auto} residuals...")
                            lstm_resid_auto_model, lstm_resid_auto_scaler, lstm_resid_auto_history = lstm.train_lstm(
                                train_residuals_auto, val_residuals_auto # Pass residuals
                            )

                            if lstm_resid_auto_model and lstm_resid_auto_scaler:
                                trained_models[model_name_hybrid_auto] = {'arima': base_auto_arima_model, 'lstm': lstm_resid_auto_model, 'scaler': lstm_resid_auto_scaler}
                                if lstm_resid_auto_history: plot_loss_curves(lstm_resid_auto_history, f"{model_name_hybrid_auto} Residual LSTM")

                                # Rolling Forecast - Use the fresh Auto ARIMA model instance
                                hybrid_auto_preds_rolling = hybrid.hybrid_rolling_forecast(train_val_series, test_series, auto_arima_model_for_hybrid_rolling, lstm_resid_auto_model, lstm_resid_auto_scaler)
                                if hybrid_auto_preds_rolling is not None:
                                    results['Rolling']['Predictions'][model_name_hybrid_auto] = hybrid_auto_preds_rolling
                                    metrics_rolling = evaluate_performance(f"{model_name_hybrid_auto} (Rolling)", test_series.values, hybrid_auto_preds_rolling.values)
                                    results['Rolling']['Metrics'][model_name_hybrid_auto] = metrics_rolling
                                else:
                                    print(f"{model_name_hybrid_auto} Rolling Forecast failed.")
                                    results['Rolling']['Metrics'][model_name_hybrid_auto] = {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

                                # Trajectory Forecast (use original trained Auto ARIMA)
                                hybrid_auto_preds_trajectory = hybrid.hybrid_trajectory_forecast(train_val_series, test_series, base_auto_arima_model, lstm_resid_auto_model, lstm_resid_auto_scaler)
                                if hybrid_auto_preds_trajectory is not None:
                                    results['Trajectory']['Predictions'][model_name_hybrid_auto] = hybrid_auto_preds_trajectory
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

    # Plot Rolling Forecasts
    plot_predictions(test_series, results['Rolling']['Predictions'], title_suffix="Rolling Forecast")

    # Plot Trajectory Forecasts
    plot_predictions(test_series, results['Trajectory']['Predictions'], title_suffix="Trajectory Forecast")

    # Display Metrics Tables
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