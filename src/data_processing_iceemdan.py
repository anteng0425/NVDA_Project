# -*- coding: utf-8 -*-
"""
Functions for processing data using ICEEMDAN decomposition and preparing
inputs for the Seq2Seq LSTM model.
"""
import numpy as np
import pandas as pd
# Use CEEMDAN
# Removed direct static import of CEEMDAN
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from functools import partial
from importlib import import_module # For dynamic import

try:
    from . import config
except ImportError:
    import config

# Helper function to dynamically get the CEEMDAN class as per GPT suggestion
def _get_ceemdan_cls():
    """
    Dynamically imports and returns the CEEMDAN class from PyEMD.CEEMDAN module.
    Ensures the actual class is retrieved, not a module object.
    """
    try:
        # Attempt to get CEEMDAN class directly if PyEMD.__init__ exports it (newer versions)
        ceemdan_module = import_module("PyEMD")
        if hasattr(ceemdan_module, "CEEMDAN") and callable(getattr(ceemdan_module, "CEEMDAN")):
            return getattr(ceemdan_module, "CEEMDAN")
    except ImportError:
        pass # Fall through to specific module import
    
    # Fallback or primary method: get from PyEMD.CEEMDAN submodule
    try:
        ceemdan_submodule = import_module("PyEMD.CEEMDAN")
        return getattr(ceemdan_submodule, "CEEMDAN")
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not dynamically import CEEMDAN class from PyEMD or PyEMD.CEEMDAN: {e}")


# Define the wrapper function at the module level so it can be pickled by joblib
def process_window_wrapper(window_data_item, trials_param, epsilon_param, target_k_param, W_param, pad_length_param, process_func):
    """
    Top-level worker function that can be pickled by joblib.
    Initializes CEEMDAN internally using dynamically retrieved class
    and calls the processing function.
    """
    CEEMDAN_class = _get_ceemdan_cls()
    try:
        # Try with epsilon (newer PyEMD versions)
        local_ceemdan_instance = CEEMDAN_class(trials=trials_param, epsilon=epsilon_param)
    except TypeError:
        # Fallback to noise_width (older PyEMD versions might expect this)
        print(f"Warning: CEEMDAN initialization with 'epsilon={epsilon_param}' failed. Trying 'noise_width={epsilon_param}'. Ensure config.ICEEMDAN_EPSILON is appropriate.")
        try:
            local_ceemdan_instance = CEEMDAN_class(trials=trials_param, noise_width=epsilon_param)
        except Exception as e_nw:
            print(f"Error: CEEMDAN initialization failed with both 'epsilon' and 'noise_width': {e_nw}")
            # Return a shape that will be filtered out by the main processing loop
            return np.array([]) # Or raise the error to stop processing

    return process_func(window_data_item, local_ceemdan_instance, target_k_param, W_param, pad_length_param)

def mirror_pad(signal, pad_length=config.ICEEMDAN_PAD_LENGTH):
    """
    Applies mirror padding to the start and end of a signal.

    Args:
        signal (np.ndarray): The 1D input signal.
        pad_length (int): The number of points to mirror pad on each side.

    Returns:
        np.ndarray: The padded signal.
    """
    if pad_length == 0:
        return signal
    # Ensure signal is 1D
    if signal.ndim != 1:
        raise ValueError("Signal for mirror_pad must be 1D.")
    if len(signal) <= pad_length:
        # If signal is too short, pad with reversed signal or zeros
        # For simplicity, let's just return a tiled version if it's extremely short,
        # though this case should ideally be handled by ensuring W is large enough.
        print(f"Warning: Signal length ({len(signal)}) is less than or equal to pad_length ({pad_length}). Padding behavior might be suboptimal.")
        if len(signal) == 0:
            return np.zeros(2 * pad_length) # Or handle as error
        return np.r_[signal[::-1][-pad_length:], signal, signal[::-1][:pad_length]]


    # Corrected padding logic:
    # For start: signal[pad_length-1::-1] or signal[pad_length-1:None:-1]
    # For end: signal[-2:-pad_length-2:-1] is complex. Simpler: signal[len(signal)-2:len(signal)-pad_length-2:-1]
    # Corrected padding logic as per GPT suggestion:
    start_pad = signal[:pad_length][::-1]  # Corrected start_pad slicing
    end_pad   = signal[-pad_length:][::-1] # end_pad was already correct based on previous GPT
    return np.r_[start_pad, signal, end_pad]


def calculate_mean_period_for_imf(imf):
    """
    Estimates the mean period of an IMF, e.g., by counting zero crossings.
    A higher number of zero crossings implies a higher frequency (shorter period).

    Args:
        imf (np.ndarray): A single IMF time series.

    Returns:
        float: An indicator of the period (e.g., inverse of zero crossing rate).
               Returns a large number if no zero crossings to ensure it's sorted as low frequency.
    """
    # Zero crossings
    crossings = np.where(np.diff(np.sign(imf)))[0]
    if len(crossings) == 0:
        return float('inf') # Effectively lowest frequency if no crossings
    # Average distance between crossings can be a proxy for period
    # Or simply use number of crossings as inverse proxy for period
    return len(imf) / (len(crossings) + 1e-6) # Add epsilon to avoid division by zero

def sort_imfs_by_frequency(imfs):
    """
    Sorts IMFs by their estimated frequency (low frequency first).

    Args:
        imfs (np.ndarray): Array of IMFs, shape (num_imfs, window_length).

    Returns:
        np.ndarray: IMFs sorted by frequency (lowest first).
    """
    if imfs.ndim == 1: # Single IMF, no sorting needed
        return imfs.reshape(1, -1)
    if imfs.shape[0] == 0: # No IMFs
        return imfs
        
    mean_periods = [calculate_mean_period_for_imf(imf) for imf in imfs]
    # Sort by period (ascending), which means frequency is descending.
    # We want low frequency first, so sort by period descending.
    # Or, if using zero crossings directly (higher is higher freq), sort ascending.
    # With current calculate_mean_period_for_imf, higher value means lower frequency.
    # So we sort in ascending order of mean_period to get low freq first.
    sorted_indices = np.argsort(mean_periods) 
    return imfs[sorted_indices]

# Renaming function slightly to align with GPT's example, though not strictly necessary
# if the call site is also updated. Keeping current name for now.
# Parameters updated to match GPT's worker example for clarity if we adopt that structure.
def process_single_window_iceemdan(window_data, local_ceemdan_instance, target_k, W, pad_length):
    """
    Processes a single window of data: mirror padding, CEEMDAN, truncation,
    IMF sorting, and fixing IMF count to K.

    Args:
        window_data (np.ndarray): Single window of raw price data (1D, length W).
        local_ceemdan_instance (CEEMDAN): Locally initialized CEEMDAN object for the worker.
        target_k (int): The target number of IMFs (e.g., config.ICEEMDAN_TARGET_K_IMF).
        W (int): Original window length.
        pad_length (int): Padding length used for mirror padding.

    Returns:
        np.ndarray: Processed data of shape (target_k + 1, W), or None if error.
    """
    # 1. Mirror Padding
    padded_window = mirror_pad(window_data, pad_length)

    # 2. CEEMDAN Execution
    try:
        # According to PyEMD documentation, calling the instance or its .ceemdan method
        # returns all components (IMFs + residual as the last one).
        all_components = local_ceemdan_instance.ceemdan(padded_window)
        # Or: all_components = local_ceemdan_instance(padded_window) - both are equivalent per docs.

        if not isinstance(all_components, np.ndarray) or all_components.ndim != 2 or all_components.shape[0] < 1:
            print(f"Warning: CEEMDAN decomposition did not return a valid 2D array of components (shape: {all_components.shape if isinstance(all_components, np.ndarray) else 'Not an ndarray'}).")
            return np.zeros((target_k + 1, W))

        # Separate IMFs and residual (residual is the last component)
        imfs = all_components[:-1]
        residual = all_components[-1]

        # Additional validation for imfs and residual shapes if necessary
        if imfs.ndim == 2 and imfs.shape[1] != padded_window.shape[0]: # Check width of IMFs
             print(f"Warning: IMFs width mismatch. Expected {padded_window.shape[0]}, got {imfs.shape[1]}.")
             return np.zeros((target_k + 1, W))
        if residual.ndim == 1 and residual.shape[0] != padded_window.shape[0]: # Check length of residual
             print(f"Warning: Residual length mismatch. Expected {padded_window.shape[0]}, got {residual.shape[0]}.")
             return np.zeros((target_k + 1, W))


    except Exception as e:
        print(f"Error during CEEMDAN decomposition for a window: {e}") # Changed text
        # Return a placeholder of zeros to maintain structure in parallel processing
        return np.zeros((target_k + 1, W))

    # 3. Truncate back to original window length W (symmetrically)
    # Padded length was W + 2*pad_length. Components have this length.
    # We need to take W points from the center.
    start_index = pad_length
    end_index = start_index + W
    
    imfs_truncated = imfs[:, start_index:end_index]
    residual_truncated = residual[start_index:end_index]

    if imfs_truncated.shape[1] != W or len(residual_truncated) != W:
        print(f"Warning: Truncation error. Expected length W={W}, got IMFs {imfs_truncated.shape[1]}, residual {len(residual_truncated)}")
        # Fallback to zeros if truncation is problematic
        return np.zeros((target_k + 1, W))

    # 4. Sort IMFs by frequency (low frequency first)
    if imfs_truncated.shape[0] > 0:
        imfs_sorted = sort_imfs_by_frequency(imfs_truncated)
    else:
        imfs_sorted = np.empty((0, W))


    # 5. Fix IMF count to K
    num_actual_imfs = imfs_sorted.shape[0]
    final_components = np.zeros((target_k + 1, W))

    if num_actual_imfs == target_k:
        final_components[:target_k, :] = imfs_sorted
        final_components[target_k, :] = residual_truncated
    elif num_actual_imfs > target_k:
        # Take first K (lowest frequency) IMFs
        final_components[:target_k, :] = imfs_sorted[:target_k, :]
        # Combine remaining IMFs with the residual
        combined_residual = imfs_sorted[target_k:, :].sum(axis=0) + residual_truncated
        final_components[target_k, :] = combined_residual
    else: # num_actual_imfs < target_k
        if num_actual_imfs > 0:
            final_components[:num_actual_imfs, :] = imfs_sorted
        # Remaining IMFs (up to target_k) are already zeros (due to initialization)
        # The residual is placed at the (target_k)-th index (last component)
        final_components[target_k, :] = residual_truncated
        
    return final_components


def generate_and_save_processed_data(full_raw_price_series, strict_train_series, force_reprocess=False):
    """
    Main function to generate all windowed IMF data using parallel processing,
    calculate global scalers (fit on strict_train_series), scale the data,
    create Seq2Seq datasets, and save/load processed data and scalers.

    Args:
        full_raw_price_series (pd.Series): The full raw stock price time series.
        strict_train_series (pd.Series): The raw stock price time series for the training set only (used for fitting scalers).
        force_reprocess (bool): If True, reprocess data even if saved files exist.

    Returns:
        tuple: (X_encoder, Y_decoder_input, Y_decoder_target, global_imf_scalers, global_price_scaler)
               or None if processing fails.
    """
    print("\n[ICEEMDAN Data Processing] Starting...")

    processed_data_path = os.path.join(config.PROCESSED_DATA_DIR, 'iceemdan_processed_data.npz')
    imf_scalers_path = os.path.join(config.SCALERS_DIR, 'iceemdan_imf_scalers.gz')
    price_scaler_path = os.path.join(config.SCALERS_DIR, 'iceemdan_price_scaler.gz')

    if not force_reprocess and \
       os.path.exists(processed_data_path) and \
       os.path.exists(imf_scalers_path) and \
       os.path.exists(price_scaler_path):
        try:
            print("[ICEEMDAN Data Processing] Loading pre-processed data and scalers...")
            data_npz = np.load(processed_data_path)
            X_encoder = data_npz['X_encoder']
            Y_decoder_input = data_npz['Y_decoder_input']
            Y_decoder_target = data_npz['Y_decoder_target']
            global_imf_scalers = joblib.load(imf_scalers_path)
            global_price_scaler = joblib.load(price_scaler_path)
            print("[ICEEMDAN Data Processing] Successfully loaded pre-processed data and scalers.")
            return X_encoder, Y_decoder_input, Y_decoder_target, global_imf_scalers, global_price_scaler
        except Exception as e:
            print(f"[ICEEMDAN Data Processing] Error loading pre-processed data: {e}. Reprocessing...")

    # --- Parameters from config ---
    W = config.ICEEMDAN_W_WINDOW_SIZE
    H = config.ICEEMDAN_H_FORECAST_PERIOD
    pad_length = config.ICEEMDAN_PAD_LENGTH
    trials = config.ICEEMDAN_TRIALS
    epsilon_value = config.ICEEMDAN_EPSILON # Ensure this uses the correct config name
    target_k = config.ICEEMDAN_TARGET_K_IMF
    encoder_timesteps = config.SEQ2SEQ_ENCODER_TIMESTEPS

    # 1. Create sliding windows from full_raw_price_series
    windows = []
    # Ensure we have enough data for at least one window and its forecast period H
    if len(full_raw_price_series) < W + H:
        print(f"Error: Not enough data in full_raw_price_series (length {len(full_raw_price_series)}) to create at least one window of size {W} and forecast horizon {H}.")
        return None, None, None, None, None

    for i in range(len(full_raw_price_series) - W - H + 1):
        windows.append(full_raw_price_series.iloc[i:i+W].values)
    
    if not windows: # Should be caught by the check above, but as a safeguard
        print("Error: Not enough data to create sliding windows from full_raw_price_series.")
        return None, None, None, None, None
    
    print(f"[ICEEMDAN Data Processing] Created {len(windows)} sliding windows of length {W} from full series.")

    # 2. Initialize CEEMDAN - This instance is not used by joblib worker, can be removed or commented.
    # ceemdan_decomposer = CEEMDAN(trials=trials, epsilon=epsilon_value)

    # 3. Process all windows in parallel to get decomposed IMFs (based on full_raw_price_series)
    print(f"[CEEMDAN Data Processing] Starting CEEMDAN decomposition for {len(windows)} windows (parallel)...")
    # Ensure process_single_window_iceemdan is picklable for joblib
    # It should be, as it only uses standard types and a pre-initialized (but picklable) CEEMDAN object.

    # Note: PyEMD objects might not be directly picklable for joblib if they contain complex state
    # or C extensions not handled by pickle. If issues arise, might need to initialize CEEMDAN
    # within the parallelized function, or use a different parallelization strategy.
    # For now, assuming it works or can be made to work.
    # A safer approach for joblib with complex objects is to pass parameters and initialize inside.
    # However, initializing CEEMDAN (which can be slow) for every window is inefficient. (This comment is now less relevant as we do it in worker)

    # Use the top-level wrapper with functools.partial
    # Parameters for CEEMDAN and processing are fixed using partial
    worker_with_fixed_params = partial(process_window_wrapper,
                                       trials_param=trials,
                                       epsilon_param=epsilon_value,
                                       target_k_param=target_k,
                                       W_param=W,
                                       pad_length_param=pad_length,
                                       process_func=process_single_window_iceemdan)

    all_windows_decomposed_unscaled = joblib.Parallel(
        n_jobs=-1, verbose=10 # Using default backend (loky) as per GPT suggestion
    )(joblib.delayed(worker_with_fixed_params)(win) for win in windows)

    # Filter out results that are not of the expected shape (e.g., from init errors in worker)
    all_windows_decomposed_unscaled = [
        res for res in all_windows_decomposed_unscaled
        if isinstance(res, np.ndarray) and res.shape == (target_k + 1, W)
    ]
    
    if not all_windows_decomposed_unscaled:
        print("Error: All window processing failed or returned incorrect shapes during CEEMDAN decomposition.")
        return None, None, None, None, None
        
    # Convert list of (K+1, W) arrays to a 3D numpy array using np.stack (GPT suggestion #3)
    all_windows_decomposed_unscaled_np = np.stack(all_windows_decomposed_unscaled)
    print(f"[CEEMDAN Data Processing] CEEMDAN decomposition complete. Stacked shape: {all_windows_decomposed_unscaled_np.shape}")

    # 4. Calculate and apply global MinMax scaling
    # For IMF/Residual components
    global_imf_scalers = []
    all_windows_decomposed_scaled_np = np.empty_like(all_windows_decomposed_unscaled_np)
    
    # Determine the number of windows that can be formed from the strict_train_series
    # This is used to fit the IMF scalers
    num_train_windows_for_scaler_fit = 0
    if len(strict_train_series) >= W:
        num_train_windows_for_scaler_fit = len(strict_train_series) - W + 1
    else:
        print(f"Warning: strict_train_series (length {len(strict_train_series)}) is shorter than window size W ({W}). Cannot fit IMF scalers properly. Scalers will be fit on potentially very little or no data from training windows.")
        # Fallback or error handling might be needed here if this is critical.
        # For now, if num_train_windows_for_scaler_fit is 0, fit will fail or be on empty data.

    for i in range(target_k + 1): # For each of the K+1 components
        scaler = MinMaxScaler(feature_range=(0, 1), clip=False) # Use clip=False
        if num_train_windows_for_scaler_fit > 0:
            # Extract component data ONLY from training windows for fitting
            component_data_train_windows = all_windows_decomposed_unscaled_np[:num_train_windows_for_scaler_fit, i, :].reshape(-1, 1)
            if component_data_train_windows.size > 0:
                scaler.fit(component_data_train_windows)
            else:
                print(f"Warning: Component {i} has no data from training windows for fitting scaler. Scaler will not be properly fitted.")
        else:
            # If no training windows, fit on a tiny, arbitrary non-empty array to avoid error, though scaler will be meaningless.
            # Or, better, handle this case by not scaling or raising an error earlier.
            # For now, to prevent fit error on empty array:
            print(f"Warning: No training windows to fit scaler for component {i}. Using a dummy fit. This scaler will not be meaningful.")
            scaler.fit(np.array([[0],[1]])) # Dummy fit to prevent error on empty data

        global_imf_scalers.append(scaler)
        
        # Apply scaling to ALL windows' component data using the scaler FIT ON TRAINING DATA
        for win_idx in range(all_windows_decomposed_unscaled_np.shape[0]):
            component_win_data = all_windows_decomposed_unscaled_np[win_idx, i, :].reshape(-1, 1)
            all_windows_decomposed_scaled_np[win_idx, i, :] = scaler.transform(component_win_data).ravel()
            
    print("[ICEEMDAN Data Processing] MinMax scaling (fit on train, clip=False) applied to IMF/Residual components.")
    joblib.dump(global_imf_scalers, imf_scalers_path)
    print(f"[ICEEMDAN Data Processing] IMF scalers saved to {imf_scalers_path}")

    # For original price data (used for Decoder targets and first input)
    # Fit on strict_train_series, then transform full_raw_price_series
    global_price_scaler = MinMaxScaler(feature_range=(0, 1), clip=False)
    global_price_scaler.fit(strict_train_series.values.reshape(-1,1))
    # Transform the full series using the scaler fitted on the training part
    scaled_full_raw_prices = global_price_scaler.transform(full_raw_price_series.values.reshape(-1,1)).flatten()
    joblib.dump(global_price_scaler, price_scaler_path)
    print(f"[ICEEMDAN Data Processing] Global price scaler (fit on train, clip=False) saved to {price_scaler_path}")

    # 5. Create Seq2Seq datasets
    # Y_decoder_input and Y_decoder_target are derived from scaled_full_raw_prices
    
    X_encoder_list = []
    Y_decoder_input_list = [] # For teacher forcing: y_T_scaled, y_T+1_scaled, ..., y_T+H-1_scaled
    Y_decoder_target_list = []  # Actual targets: y_T+1_scaled, ..., y_T+H_scaled

    for i in range(len(windows)): # Iterate up to the number of successfully processed windows
        # Encoder input: last `encoder_timesteps` of the scaled IMFs for window i
        # all_windows_decomposed_scaled_np shape: (num_windows, K+1, W)
        # We need to transpose to (num_windows, W, K+1) for easier slicing of timesteps
        scaled_imfs_for_window_i_transposed = all_windows_decomposed_scaled_np[i].T # Shape (W, K+1)
        X_encoder_list.append(scaled_imfs_for_window_i_transposed[-encoder_timesteps:, :])

        # Decoder target: next H prices after window i, scaled
        # Window i covers raw_price_series from index `i` to `i+W-1`
        # Target starts from `i+W` up to `i+W+H-1`
        target_start_idx = i + W
        target_end_idx = target_start_idx + H
        # Get from the globally scaled raw prices
        raw_target_sequence_scaled = scaled_full_raw_prices[target_start_idx:target_end_idx]
        Y_decoder_target_list.append(raw_target_sequence_scaled.reshape(H, 1))

        # Decoder input for teacher forcing:
        # Starts with y_T_scaled (price at the end of window W for current sample i)
        # then y_T+1_scaled, ..., y_T+H-1_scaled
        # y_T_scaled is scaled_full_raw_prices[i+W-1]
        # y_T+j_scaled is scaled_full_raw_prices[i+W-1+j]
        decoder_input_seq_scaled = np.zeros((H, 1))
        decoder_input_seq_scaled[0,0] = scaled_full_raw_prices[i + W - 1] # y_T_scaled
        if H > 1:
            decoder_input_seq_scaled[1:,0] = raw_target_sequence_scaled[:-1] # y_T+1_scaled to y_T+H-1_scaled
        Y_decoder_input_list.append(decoder_input_seq_scaled)

    X_encoder = np.array(X_encoder_list)
    Y_decoder_input = np.array(Y_decoder_input_list)
    Y_decoder_target = np.array(Y_decoder_target_list)
    
    print(f"[ICEEMDAN Data Processing] Seq2Seq datasets created. Shapes: X_enc={X_encoder.shape}, Y_dec_in={Y_decoder_input.shape}, Y_dec_tgt={Y_decoder_target.shape}")

    # Save processed data
    np.savez_compressed(processed_data_path, X_encoder=X_encoder, Y_decoder_input=Y_decoder_input, Y_decoder_target=Y_decoder_target)
    print(f"[ICEEMDAN Data Processing] Processed data saved to {processed_data_path}")
    
    return X_encoder, Y_decoder_input, Y_decoder_target, global_imf_scalers, global_price_scaler

if __name__ == '__main__':
    # Example Usage (requires a CSV file at config.CSV_PATH)
    print("Testing data_processing_iceemdan.py...")
    # Load raw data (example)
    try:
        raw_df = pd.read_csv(config.CSV_PATH, index_col='Date', parse_dates=True)
        # Ensure 'adj_close' is present and use it
        if 'adj_close' not in raw_df.columns and 'Adj Close' in raw_df.columns:
            raw_df['adj_close'] = raw_df['Adj Close']
        
        if 'adj_close' in raw_df:
            price_series = raw_df['adj_close'].dropna()
            
            # Filter by cutoff date if needed, similar to main data_processing
            price_series = price_series[price_series.index < pd.to_datetime(config.CUTOFF_DATE)]


            if len(price_series) > config.ICEEMDAN_W_WINDOW_SIZE + config.ICEEMDAN_H_FORECAST_PERIOD:
                print(f"Using price series of length {len(price_series)}")
                # For testing, we need a dummy strict_train_series. Let's take the first 70% of price_series.
                # This is just for the __main__ test block, not for the actual pipeline.
                # In the actual pipeline, strict_train_series will be train_series_orig.
                num_strict_train_points = int(len(price_series) * 0.7) # Example split for testing
                if num_strict_train_points < config.ICEEMDAN_W_WINDOW_SIZE : # Ensure enough for at least one window for scaler fitting
                    print(f"Warning: Test strict_train_series (length {num_strict_train_points}) too short for window size {config.ICEEMDAN_W_WINDOW_SIZE}. Adjusting for test.")
                    num_strict_train_points = config.ICEEMDAN_W_WINDOW_SIZE
                
                if len(price_series) >= num_strict_train_points and num_strict_train_points > 0:
                    test_strict_train_series = price_series.iloc[:num_strict_train_points]
                    print(f"Using test_strict_train_series of length {len(test_strict_train_series)} for scaler fitting in test.")
                    X_enc, Y_dec_in, Y_dec_tgt, imf_scalers, price_scaler = generate_and_save_processed_data(
                        full_raw_price_series=price_series,
                        strict_train_series=test_strict_train_series,
                        force_reprocess=True
                    )
                    if X_enc is not None:
                        print("Processing finished.")
                else:
                    print("Not enough data to form a meaningful strict_train_series for testing. Skipping processing.")
                    X_enc, Y_dec_in, Y_dec_tgt, imf_scalers, price_scaler = [None]*5

                if X_enc is not None: # Check again after potential skip
                    print("Processing finished (or skipped due to insufficient test data for strict_train_series).")
                    print(f"X_encoder sample shape: {X_enc[0].shape if len(X_enc) > 0 else 'N/A'}")
                    print(f"Y_decoder_input sample shape: {Y_dec_in[0].shape if len(Y_dec_in) > 0 else 'N/A'}")
                    print(f"Y_decoder_target sample shape: {Y_dec_tgt[0].shape if len(Y_dec_tgt) > 0 else 'N/A'}")
                    print(f"Number of IMF/Res scalers: {len(imf_scalers)}")
                    print(f"Price scaler min: {price_scaler.min_}, max: {price_scaler.data_max_}")
                else:
                    print("Processing failed.")
            else:
                print(f"Not enough data after filtering for ICEEMDAN processing. Min length required: {config.ICEEMDAN_W_WINDOW_SIZE + config.ICEEMDAN_H_FORECAST_PERIOD +1}, available: {len(price_series)}")
        else:
            print("Error: 'adj_close' column not found in CSV for testing.")
            
    except FileNotFoundError:
        print(f"Error: Test CSV file not found at {config.CSV_PATH}")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()