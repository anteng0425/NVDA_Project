# -*- coding: utf-8 -*-
"""
Functions for visualizing model results (loss curves, predictions).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Import constants from config (assuming config.py is in the same directory)
try:
    from . import config
except ImportError:
    import config # Fallback for direct execution or different environment setup

# Apply plot style from config
plt.style.use(config.PLOT_STYLE)

def plot_loss_curves(history, model_name):
    """Plots training and validation loss curves and saves the plot."""
    if history is None or not hasattr(history, 'history') or 'loss' not in history.history or 'val_loss' not in history.history:
        print(f"No history data to plot for {model_name}.")
        return

    print(f"\n[Visualization] Plotting loss curves for {model_name}...")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label=f'{model_name} Train Loss')
    plt.plot(history.history['val_loss'], label=f'{model_name} Validation Loss')
    plt.title(f'{model_name} Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save the plot before showing
    plot_filename = os.path.join(config.RESULTS_PLOTS_DIR, f'{model_name}_loss_curve.png') # Use updated variable name
    try:
        # Ensure directory exists (redundant if done at start, but safe)
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        plt.savefig(plot_filename)
        print(f"Loss curve saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving loss curve plot: {e}")
    plt.show() # Re-enable interactive plot window

def plot_predictions(test_series_actual, predictions_dict, title_suffix="", y_label='Adjusted Close Price'): # Added y_label parameter
    """Plots actual vs predicted values for multiple models and saves the plot."""
    print(f"\n[Visualization] Plotting predictions for {title_suffix}...")
    plt.figure(figsize=(14, 7))
    # Plot actual values with a distinct style
    plt.plot(test_series_actual.index, test_series_actual.values, label='Actual Adj Close', color='black', linewidth=2.5, linestyle='-')

    test_dates = test_series_actual.index # Get index for potential alignment

    # Define a color cycle for predictions
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] # Get default color cycle
    # Or define custom colors:
    # color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_index = 0

    # Sort models alphabetically for consistent color assignment (optional but good practice)
    sorted_model_names = sorted(predictions_dict.keys())

    for model_name in sorted_model_names:
        preds = predictions_dict[model_name]
        if preds is None: # Skip if predictions failed
             print(f"[Visualization] Skipping plot for {model_name} due to missing predictions.")
             continue

        # Assign color from cycle
        current_color = color_cycle[color_index % len(color_cycle)]
        color_index += 1

        # Handle predictions (Series or ndarray) - Use SOLID lines for all predictions
        if isinstance(preds, pd.Series):
            valid_preds = preds.dropna()
            if not valid_preds.empty:
                 plt.plot(valid_preds.index, valid_preds.values, label=f'{model_name} Predicted', linestyle='-', color=current_color, linewidth=1.5) # Solid line
            else:
                 print(f"Warning: No valid (non-NaN) Series predictions to plot for {model_name}.")
        elif isinstance(preds, np.ndarray):
             if len(preds) == len(test_dates):
                 valid_mask = ~np.isnan(preds)
                 if np.any(valid_mask):
                      plt.plot(test_dates[valid_mask], preds[valid_mask], label=f'{model_name} Predicted', linestyle='-', color=current_color, linewidth=1.5) # Solid line
                 else:
                      print(f"Warning: Numpy prediction array contains only NaNs for {model_name}.")
             else:
                 print(f"Warning: Numpy prediction array length mismatch for {model_name}. Cannot plot reliably.")
        else:
             try:
                 preds_array = np.array(preds)
                 if preds_array.ndim == 1 and len(preds_array) == len(test_dates):
                      valid_mask = ~np.isnan(preds_array)
                      if np.any(valid_mask):
                           plt.plot(test_dates[valid_mask], preds_array[valid_mask], label=f'{model_name} Predicted', linestyle='-', color=current_color, linewidth=1.5) # Solid line
                      else:
                           print(f"Warning: Fallback prediction array contains only NaNs for {model_name}.")
                 else:
                      print(f"Warning: Could not interpret or align prediction type ({type(preds)}) for {model_name}. Cannot plot.")
             except Exception as plot_err:
                 print(f"Warning: Error plotting fallback prediction type ({type(preds)}) for {model_name}: {plot_err}")


    plt.title(f'NVDA Stock Price Prediction {title_suffix}')
    plt.xlabel('Date')
    plt.ylabel(y_label) # Use the y_label parameter
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot before showing
    # Sanitize title_suffix for filename
    safe_suffix = "".join(c if c.isalnum() else "_" for c in title_suffix.lower())
    plot_filename = os.path.join(config.RESULTS_PLOTS_DIR, f'predictions_{safe_suffix}.png') # Use updated variable name
    try:
        # Ensure directory exists (redundant if done at start, but safe)
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        plt.savefig(plot_filename)
        print(f"Prediction plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving prediction plot: {e}")
    plt.show() # Re-enable interactive plot window

# Example of direct execution for testing (optional)
if __name__ == '__main__':
    print("Testing visualization module...")
    # Create dummy data for testing plot_predictions
    dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
    actual = pd.Series(np.random.randn(100).cumsum() + 50, index=dates)
    preds1 = actual * np.random.uniform(0.95, 1.05, size=100)
    preds2 = actual * np.random.uniform(0.90, 1.10, size=100)
    preds_dict = {"Model A": preds1, "Model B": preds2, "Failed Model": None}

    plot_predictions(actual, preds_dict, title_suffix="Test Plot")

    # Create dummy history for testing plot_loss_curves
    class DummyHistory:
        def __init__(self):
            self.history = {
                'loss': np.linspace(1, 0.1, 50) + np.random.rand(50) * 0.1,
                'val_loss': np.linspace(0.8, 0.2, 50) + np.random.rand(50) * 0.05
            }
    dummy_hist = DummyHistory()
def plot_full_history(df, title="Full Adjusted Close Price History"):
    """Plots the full preprocessed historical data and saves the plot."""
    if df is None or df.empty or 'adj_close' not in df.columns:
        print("[Visualization] Error: Invalid DataFrame provided for plotting full history.")
        return

    print(f"\n[Visualization] Plotting {title}...")
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['adj_close'], label='Adjusted Close')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot before showing
    plot_filename = os.path.join(config.RESULTS_PLOTS_DIR, 'full_history_plot.png') # Use updated variable name
    try:
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        plt.savefig(plot_filename)
        print(f"Full history plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving full history plot: {e}")
    plt.show() # Show interactive plot

    print("Visualization tests complete (check saved plots).")