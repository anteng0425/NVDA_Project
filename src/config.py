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
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_iceemdan') # For ICEEMDAN processed data
MODEL_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'models', 'saved_weights') # For saving model weights
SCALERS_DIR = os.path.join(PROJECT_ROOT, 'models', 'saved_scalers') # For saving scalers

# Ensure directories exist
os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
os.makedirs(SCALERS_DIR, exist_ok=True)

# --- Training Device Configuration ---
# Set to True to force TensorFlow to use CPU even if GPU is available.
# Set to False to allow TensorFlow to automatically use GPU if available.
FORCE_CPU_TRAINING = True


# --- Data Constants ---
CSV_PATH = os.path.join(DATA_DIR, 'NVDA_stock_data_new.csv')
CUTOFF_DATE = '2023-03-14'
TRAIN_VAL_RATIO = 0.8
TRAIN_RATIO = 0.8

# --- LSTM Model Hyperparameters (Shared) ---
LSTM_WINDOW_SIZE = 20 # Increased window size
# Dropout rates for LSTM layers:
# LSTM_STANDARD_DROPOUT_RATE applies to the non-recurrent connections (input/output gates of each timestep).
# LSTM_RECURRENT_DROPOUT_RATE applies to the recurrent connections (between timesteps).
LSTM_STANDARD_DROPOUT_RATE = 0.0 # Example: Can be adjusted
LSTM_RECURRENT_DROPOUT_RATE = 0.2 # Example: Can be adjusted

# --- Pure LSTM Specific Hyperparameters ---
PURE_LSTM_EPOCHS = 500 # Increased epochs
PURE_LSTM_BATCH_SIZE = 64 # Halved batch size again
PURE_LSTM_PATIENCE = 300 # Increased patience
PURE_LSTM_UNITS_1 = 32 # Increased units in first LSTM layer
PURE_LSTM_UNITS_2 = 16 # Increased units in second LSTM layer
PURE_LSTM_DENSE_UNITS = 16 # Increased units in Dense layer
# Pure LSTM will use tanh activation and recurrent dropout (handled in model creation)

# --- Pure LSTM N-Step Teacher Forcing Specific Hyperparameters ---
# The following N-Step Teacher Forcing LSTM model was an experiment to improve trajectory prediction.
# However, it resulted in poor performance, with predictions appearing as a significantly
# time-shifted version of the actual trajectory. Due to these issues, these configurations are commented out.
# N_STEP_AHEAD_TF = 5 # Number of steps ahead to train with teacher forcing
# PURE_LSTM_NSTEP_TF_EPOCHS = 300 # Can be adjusted
# PURE_LSTM_NSTEP_TF_BATCH_SIZE = 32 # Can be adjusted
# PURE_LSTM_NSTEP_TF_PATIENCE = 150 # Can be adjusted
# PURE_LSTM_NSTEP_TF_UNITS_1 = 64 # Same as Pure LSTM
# PURE_LSTM_NSTEP_TF_UNITS_2 = 32 # Same as Pure LSTM
# PURE_LSTM_NSTEP_TF_DENSE_UNITS = 32 # Same as Pure LSTM
# # This model will also use tanh activation and recurrent dropout, and be bidirectional

# --- Hybrid (Residual) LSTM Specific Hyperparameters ---
HYBRID_LSTM_EPOCHS = 300 # Original value
HYBRID_LSTM_BATCH_SIZE = 16 # Original value
HYBRID_LSTM_PATIENCE = 150 # Original value
HYBRID_LSTM_UNITS_1 = 16 # Original value
HYBRID_LSTM_UNITS_2 = 8 # Original value
HYBRID_LSTM_DENSE_UNITS = 8 # Original value
# Hybrid LSTM will use relu activation and standard dropout (handled in model creation)
HYBRID_LSTM_APPLY_SCALING_TO_RESIDUALS = True # Controls if MinMaxScaler is applied to ARIMA residuals before Hybrid LSTM training
HYBRID_USE_LOG_TRANSFORM = False # If True, Hybrid models operate on log-transformed prices. If False, on original prices.
# --- Auto ARIMA Parameters ---
AUTO_ARIMA_START_P = 0
AUTO_ARIMA_START_Q = 0
AUTO_ARIMA_MAX_P = 5
AUTO_ARIMA_MAX_Q = 5
AUTO_ARIMA_SEASONAL = False
AUTO_ARIMA_TRACE = False # Set to True for detailed search steps

# --- CEEMDAN/ICEEMDAN - Seq2Seq LSTM Configuration ---
ICEEMDAN_W_WINDOW_SIZE = 378       # Large window for ICEEMDAN decomposition
ICEEMDAN_H_FORECAST_PERIOD = 5     # How many steps ahead to predict
ICEEMDAN_PAD_LENGTH = 30           # Mirror padding length for boundary effects
ICEEMDAN_TRIALS = 200              # Number of trials for ICEEMDAN
ICEEMDAN_EPSILON = 0.2             # Epsilon for CEEMDAN (noise scale related to std)
ICEEMDAN_TARGET_K_IMF = 8          # Target number of IMFs (excluding residual)

SEQ2SEQ_ENCODER_TIMESTEPS = 10     # Timesteps for Encoder input (taken from end of W_WINDOW_SIZE IMFs)
SEQ2SEQ_ENCODER_LSTM_UNITS = [96, 96] # Units for Encoder LSTM layers (e.g., [layer1_units, layer2_units])
SEQ2SEQ_DECODER_LSTM_UNITS = [96, 96] # Units for Decoder LSTM layers
# SEQ2SEQ_DENSE_OUTPUT_UNITS = 1 # Implicitly 1 for single value prediction per step
SEQ2SEQ_LEARNING_RATE = 0.001
SEQ2SEQ_BATCH_SIZE = 144
SEQ2SEQ_EPOCHS = 1000 # Changed to 1000 as requested
SEQ2SEQ_PATIENCE = 300  # Changed to 200 as requested
# Standard and Recurrent Dropout rates for Seq2Seq LSTMs will use the shared
# LSTM_STANDARD_DROPOUT_RATE and LSTM_RECURRENT_DROPOUT_RATE by default,
# but can be overridden if specific config values are added for Seq2Seq.
# For now, we assume they use the shared ones.
SEQ2SEQ_USE_ATTENTION = True # Flag to enable/disable Attention mechanism in Decoder

# --- ICEEMDAN Model Loading/Saving Strategy ---
ICEEMDAN_LOAD_SAVED_MODEL = False     # If True, try to load a pre-trained model.
ICEEMDAN_FORCE_RETRAIN = True       # If True, always retrain even if a saved model exists that could be loaded.
ICEEMDAN_MODEL_VERSION_TO_LOAD = "latest" # "latest", a specific version ID (e.g., "20230516-123000"), or "" to prompt (if implemented).
ICEEMDAN_BASE_SAVE_DIR = os.path.join(MODEL_WEIGHTS_DIR, "ICEEMDAN_Seq2Seq_LSTM_versions") # Base directory for versioned models

# --- ICEEMDAN Data Processing Strategy ---
ICEEMDAN_DATA_FORCE_REPROCESS = True # If True, force reprocessing of ICEEMDAN data (IMFs, scalers, npz file), ignoring existing files. Set to True once after scaler logic changes.

# --- Plotting Configuration ---
PLOT_STYLE = 'fivethirtyeight'