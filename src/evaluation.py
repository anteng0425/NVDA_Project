# -*- coding: utf-8 -*-
"""
Functions for evaluating the performance of forecasting models.
"""

import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score

def calculate_rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error (RMSE)."""
    return sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Filter out zero values in y_true to avoid division by zero
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        print("Warning: No non-zero true values for MAPE calculation. Returning inf.")
        return np.inf
    y_true_filt = y_true[non_zero_mask]
    y_pred_filt = y_pred[non_zero_mask]
    return np.mean(np.abs((y_true_filt - y_pred_filt) / y_true_filt)) * 100

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
        print("Warning: Less than 2 valid data points for direction accuracy calculation. Returning 0.")
        return 0.0

    # Calculate differences on valid data
    true_diff = np.diff(y_true_valid)
    # Predict difference based on prediction and *previous prediction* (as per Markdown)
    pred_diff = np.diff(y_pred_valid) # Compare change in prediction with change in actual

    # Ensure lengths match after diff calculation
    min_len = min(len(true_diff), len(pred_diff))
    if min_len <= 0:
        print("Warning: No comparable differences for direction accuracy calculation. Returning 0.")
        return 0.0 # Return 0 if no comparable differences

    # Compare the sign of the differences
    # Handle cases where diff is zero (sign becomes 0) - consider them as incorrect direction? Or neutral?
    # Current approach: sign(0) == sign(0) is True, sign(positive) == sign(0) is False.
    correct_direction = (np.sign(true_diff[:min_len]) == np.sign(pred_diff[:min_len]))
    return np.mean(correct_direction) * 100

def calculate_r2(y_true, y_pred):
    """Calculates the R-squared (Coefficient of Determination)."""
    # Ensure there are enough points for R2 calculation
    if len(y_true) < 2:
        print("Warning: Less than 2 data points for R2 calculation. Returning NaN.")
        return np.nan
    try:
        return r2_score(y_true, y_pred)
    except ValueError as e:
        print(f"Warning: Could not calculate R2 score: {e}. Returning NaN.")
        return np.nan


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

    print(f"\n--- Evaluating {model_name} Performance ---")
    try:
        rmse = calculate_rmse(y_true_valid, y_pred_valid)
        mape = calculate_mape(y_true_valid, y_pred_valid)
        # Pass the original arrays to ACC, it handles NaNs internally now
        acc = calculate_direction_accuracy(y_true_arr, y_pred_arr)
        r2 = calculate_r2(y_true_valid, y_pred_valid)

        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.4f}%")
        print(f"Direction Accuracy (ACC): {acc:.4f}%")
        print(f"R-squared (R2): {r2:.4f}")
        print("-" * (len(model_name) + 22))
        return {"RMSE": rmse, "MAPE": mape, "ACC": acc, "R2": r2}
    except Exception as e:
        print(f"Error during evaluation for {model_name}: {e}")
        return {k: np.nan for k in ["RMSE", "MAPE", "ACC", "R2"]}

# Example of direct execution for testing (optional)
if __name__ == '__main__':
    print("Testing evaluation module...")
    y_t = np.array([10, 11, 10.5, 11.5, 12, 11])
    y_p_good = np.array([10.1, 10.9, 10.6, 11.4, 11.9, 11.1])
    y_p_bad = np.array([11, 10, 11, 10, 11, 12])
    y_p_nan = np.array([10.1, np.nan, 10.6, 11.4, np.nan, 11.1])
    y_p_zero = np.array([0, 0, 0, 0, 0, 0])

    print("\nEvaluating Good Predictions:")
    evaluate_performance("Good Model", y_t, y_p_good)

    print("\nEvaluating Bad Predictions:")
    evaluate_performance("Bad Model", y_t, y_p_bad)

    print("\nEvaluating Predictions with NaNs:")
    evaluate_performance("NaN Model", y_t, y_p_nan)

    print("\nEvaluating Zero Predictions:")
    evaluate_performance("Zero Model", y_t, y_p_zero)

    print("\nEvaluating with Zero True Value:")
    y_t_with_zero = np.array([10, 0, 10.5, 11.5, 12, 11])
    evaluate_performance("Zero True Model", y_t_with_zero, y_p_good)