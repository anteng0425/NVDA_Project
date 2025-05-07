# -*- coding: utf-8 -*-
"""
Functions for loading, preprocessing, and splitting the NVDA stock data.
"""

import pandas as pd
import os
from datetime import datetime
import warnings

# Import constants from config (assuming config.py is in the same directory)
try:
    from . import config
except ImportError:
    import config # Fallback for direct execution or different environment setup

def load_and_preprocess_data(csv_path=config.CSV_PATH, cutoff_date=config.CUTOFF_DATE):
    """
    Loads NVDA stock data from a CSV file, preprocesses it according
    to the project requirements. Sets 'date' as index.

    Args:
        csv_path (str): Path to the CSV file. Defaults to config.CSV_PATH.
        cutoff_date (str): The date (YYYY-MM-DD) before which data should be kept. Defaults to config.CUTOFF_DATE.

    Returns:
        pd.DataFrame: Preprocessed data with DatetimeIndex and 'adj_close' column,
                      sorted by date, or None if loading fails.
    """
    print(f"\n[Data Processing] Loading data from: {csv_path}")
    if not os.path.exists(csv_path):
         print(f"Error: CSV file not found at {csv_path}")
         # Try alternative path relative to project root if called directly
         alt_path = os.path.join(config.PROJECT_ROOT, 'data', 'raw', os.path.basename(csv_path))
         if os.path.exists(alt_path):
             print(f"Attempting alternative path: {alt_path}")
             csv_path = alt_path
         else:
             return None

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
            df_orig_len = len(df)
            df = df.asfreq('B')
            print("Set DataFrame frequency to 'B' (Business Day).")
            # Check if resampling introduced NaNs and forward fill them
            if df['adj_close'].isnull().any():
                nan_count = df['adj_close'].isnull().sum()
                if nan_count > 0:
                    print(f"Warning: {nan_count} NaNs introduced after setting frequency to 'B'. Forward filling...")
                    # Use interpolate instead of ffill for potentially better results
                    df['adj_close'].interpolate(method='time', inplace=True)
                    # df['adj_close'].fillna(method='ffill', inplace=True) # Original ffill
                    # Drop any remaining NaNs at the very beginning if interpolate couldn't fill them
                    df.dropna(inplace=True)
                    print(f"NaNs filled/dropped. Final shape after frequency setting: {df.shape}")
        except ValueError as freq_error:
             # This might happen if dates are not unique or other index issues
             print(f"Warning: Could not set frequency to 'B'. Error: {freq_error}. ARIMA models might fail.")
             # Proceed without frequency if setting fails

        return df

    except FileNotFoundError:
        # This case should be caught by the initial os.path.exists check, but included for safety
        print(f"Error: CSV file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading/preprocessing: {e}")
        return None

def split_data(df, train_val_ratio=config.TRAIN_VAL_RATIO, train_ratio=config.TRAIN_RATIO):
    """
    Splits the dataframe (with DatetimeIndex) into training, validation,
    and test sets based on ratios.

    Args:
        df (pd.DataFrame): The preprocessed dataframe with DatetimeIndex.
        train_val_ratio (float): The proportion of data for training + validation. Defaults to config.TRAIN_VAL_RATIO.
        train_ratio (float): The proportion of the train+val set for actual training. Defaults to config.TRAIN_RATIO.

    Returns:
        tuple: Contains train_df, val_df, test_df, train_val_df (all pd.DataFrames).
               Returns (None, None, None, None) if splitting fails or df is invalid.
    """
    print("\n[Data Processing] Splitting data...")
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

# Example of direct execution for testing (optional)
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    print("Testing data_processing module...")
    loaded_df = load_and_preprocess_data()
    if loaded_df is not None:
        print("\n--- Data Head ---")
        print(loaded_df.head())
        print("\n--- Data Tail ---")
        print(loaded_df.tail())
        print("\n--- Data Info ---")
        loaded_df.info()

        tr_df, v_df, te_df, tv_df = split_data(loaded_df)
        if tr_df is not None:
            print("\nSplitting successful.")
            print(f"Train shape: {tr_df.shape}")
            print(f"Validation shape: {v_df.shape}")
            print(f"Test shape: {te_df.shape}")
        else:
            print("\nSplitting failed.")
    else:
        print("\nData loading failed.")