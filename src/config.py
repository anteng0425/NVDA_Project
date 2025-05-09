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
RESULTS_PLOTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'plots') # Renamed for clarity
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs', 'tensorboard') # Base directory for TensorBoard logs

# Ensure directories exist
os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# --- Data Constants ---
CSV_PATH = os.path.join(DATA_DIR, 'NVDA_stock_data_new.csv')
CUTOFF_DATE = '2023-03-14'
TRAIN_VAL_RATIO = 0.8
TRAIN_RATIO = 0.8

# --- LSTM Model Hyperparameters (Shared) ---
LSTM_WINDOW_SIZE = 60 # Increased window size
LSTM_DROPOUT_RATE = 0.3 # Increased dropout rate

# --- Pure LSTM Specific Hyperparameters ---
PURE_LSTM_EPOCHS = 300 # Increased epochs
PURE_LSTM_BATCH_SIZE = 32 # Halved batch size again
PURE_LSTM_PATIENCE = 150 # Increased patience
PURE_LSTM_UNITS_1 = 64 # Increased units in first LSTM layer
PURE_LSTM_UNITS_2 = 32 # Increased units in second LSTM layer
PURE_LSTM_DENSE_UNITS = 32 # Increased units in Dense layer
# Pure LSTM will use tanh activation and recurrent dropout (handled in model creation)

# --- Hybrid (Residual) LSTM Specific Hyperparameters ---
HYBRID_LSTM_EPOCHS = 300 # Original value
HYBRID_LSTM_BATCH_SIZE = 128 # Original value
HYBRID_LSTM_PATIENCE = 30 # Original value
HYBRID_LSTM_UNITS_1 = 32 # Original value
HYBRID_LSTM_UNITS_2 = 16 # Original value
HYBRID_LSTM_DENSE_UNITS = 16 # Original value
# Hybrid LSTM will use relu activation and standard dropout (handled in model creation)

# --- Auto ARIMA Parameters ---
AUTO_ARIMA_START_P = 0
AUTO_ARIMA_START_Q = 0
AUTO_ARIMA_MAX_P = 5
AUTO_ARIMA_MAX_Q = 5
AUTO_ARIMA_SEASONAL = False
AUTO_ARIMA_TRACE = False # Set to True for detailed search steps

# --- Plotting Configuration ---
PLOT_STYLE = 'fivethirtyeight'