# -*- coding: utf-8 -*-
"""
Configuration settings for the NVDA stock prediction project.
"""

import os
import sys

# --- Project Path Configuration ---
# Get the directory of the current script (config.py in src/)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for environments where __file__ is not defined
    script_dir = os.getcwd()
# Get the project root directory (one level up from src/)
PROJECT_ROOT = os.path.dirname(script_dir)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'plots')

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)


# --- Data Constants ---
CSV_PATH = os.path.join(DATA_DIR, 'NVDA_stock_data_new.csv')
CUTOFF_DATE = '2023-03-14'
TRAIN_VAL_RATIO = 0.8
TRAIN_RATIO = 0.8

# --- LSTM Model Hyperparameters ---
LSTM_WINDOW_SIZE = 20
LSTM_EPOCHS = 300 # As per documentation, though might be long
LSTM_BATCH_SIZE = 128
LSTM_PATIENCE = 30
LSTM_UNITS_1 = 32
LSTM_UNITS_2 = 16
LSTM_DENSE_UNITS = 16
LSTM_DROPOUT_RATE = 0.2

# --- Auto ARIMA Parameters ---
AUTO_ARIMA_START_P = 0
AUTO_ARIMA_START_Q = 0
AUTO_ARIMA_MAX_P = 5
AUTO_ARIMA_MAX_Q = 5
AUTO_ARIMA_SEASONAL = False
AUTO_ARIMA_TRACE = False # Set to True for detailed search steps

# --- Plotting Configuration ---
PLOT_STYLE = 'fivethirtyeight'