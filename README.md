# NVDA Stock Price Prediction Project

This project aims to predict NVIDIA (NVDA) stock prices using various time series forecasting models, including ARIMA, LSTM, and Hybrid ARIMA+LSTM approaches.

## Project Structure

```
nvda_stock_predictor/
├── data/
│   ├── raw/                  # Raw data (e.g., NVDA_stock_data_new.csv)
│   └── processed_iceemdan/   # Processed data from ICEEMDAN (e.g., iceemdan_processed_data.npz)
├── docs/                     # Documentation (Note: Potentially outdated, refer to code and this README for current details)
├── models/                   # Saved models, weights, and scalers
│   ├── saved_scalers/        # Saved scalers (e.g., iceemdan_imf_scalers.gz, iceemdan_price_scaler.gz)
│   └── saved_weights/        # Saved model weights
│       ├── ICEEMDAN_Seq2Seq_LSTM_versions/ # Versioned SavedModels and checkpoints for ICEEMDAN-Seq2Seq LSTM
│       └── ...               # Other model weights if saved directly
├── notebooks/                # (Optional) Jupyter notebooks for exploration
├── results/
│   ├── plots/                # Saved plots (loss curves, predictions)
│   └── metrics/              # (Optional) Saved metrics files
├── scripts/                  # (Optional) Utility scripts
├── src/                      # Source code package
│   ├── __init__.py           # Makes src a package
│   ├── config.py             # Configuration constants and paths
│   ├── data_processing.py    # Data loading and splitting functions for standard models
│   ├── data_processing_iceemdan.py # Data processing for ICEEMDAN and Seq2Seq model
│   ├── evaluation.py         # Model evaluation metric functions
│   ├── main.py               # Main execution script orchestrating the pipeline
│   ├── visualization.py      # Plotting functions
│   └── models/               # Model implementations sub-package
│       ├── __init__.py       # Makes models a sub-package
│       ├── arima.py
│       ├── hybrid.py
│       ├── lstm.py
│       ├── naive.py
│       └── seq2seq_attention_lstm.py # Seq2Seq LSTM with Attention model
├── logs/                     # Log files (e.g., for TensorBoard)
│   └── tensorboard/          # TensorBoard specific log files (organized by run)
├── .gitignore                # Files ignored by Git
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd nvda_stock_predictor
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    **Note on GPU Support & Python Version:** The `requirements.txt` file specifies `tensorflow-gpu==2.10.0`. To use GPU acceleration, you need a compatible NVIDIA GPU, drivers, CUDA Toolkit, and cuDNN. TensorFlow 2.10 is compatible with Python versions 3.7-3.10. If you lack a GPU or prefer CPU, modify `requirements.txt` to `tensorflow-cpu` and ensure your Python version is compatible. The project also uses `EMD-signal` for ICEEMDAN and `joblib` for parallel processing.

## Data

- Raw stock data: `data/raw/NVDA_stock_data_new.csv`.
- Processed ICEEMDAN data: `data/processed_iceemdan/iceemdan_processed_data.npz` (contains `X_encoder`, `Y_decoder_input`, `Y_decoder_target`).
- Saved model weights: `models/saved_weights/`. For ICEEMDAN-Seq2Seq LSTM, versioned directories (e.g., `YYYYMMDD-HHMMSS/`) store Keras SavedModels and checkpoints.
- Saved data scalers: `models/saved_scalers/` (e.g., `iceemdan_imf_scalers.gz`, `iceemdan_price_scaler.gz`).

## Usage

**Important:** Due to the use of relative imports within the `src` package, the main script should be run as a module from the **project's root directory** (i.e., the directory that contains the `nvda_stock_predictor` package directory).

1.  Navigate to the project's root directory in your termi nal.
2.  Run the following command:

    ```bash
    python -m nvda_stock_predictor.src.main
    ```

This command tells Python to execute the `main` module located inside the `nvda_stock_predictor.src` package.

The script (`src/main.py`) orchestrates the following pipeline:
1.  **Data Loading & Preprocessing:** Loads raw data from `data/raw/`, preprocesses it (handles dates, sets frequency, interpolates missing business days), and splits into training, validation, and test sets. Both original and log-transformed series are prepared.
2.  **Model Training & Prediction:**
    -   **Naive Forecast:** Rolling and trajectory.
    -   **ARIMA(1,1,1) & Auto ARIMA:** Trained on log-transformed data. Rolling and trajectory forecasts.
    -   **Pure Bi-LSTM:** Trained on original price data. LSTM layers can be configured for activation, standard/recurrent dropout. Rolling and trajectory forecasts.
    -   **Hybrid Models (ARIMA(1,1,1)+LSTM, Auto ARIMA+LSTM):**
        -   Can operate on original or log-transformed prices (configurable via `src/config.py:HYBRID_USE_LOG_TRANSFORM`).
        -   ARIMA component predicts the main trend, LSTM component models the residuals.
        -   LSTM for residuals can also be configured (activation, dropout).
        -   Rolling and trajectory forecasts.
    -   **ICEEMDAN-Seq2Seq LSTM with Attention:**
        -   Performs ICEEMDAN decomposition on the full raw price series. Data processing can be forced (via `src/config.py:ICEEMDAN_DATA_FORCE_REPROCESS`).
        -   Trains a Seq2Seq LSTM model with Bahdanau Attention on the decomposed components and price.
        -   Supports loading pre-trained models or forcing retraining (configurable via `src/config.py:ICEEMDAN_LOAD_SAVED_MODEL`, `ICEEMDAN_FORCE_RETRAIN`, `ICEEMDAN_MODEL_VERSION_TO_LOAD`).
        -   Rolling and trajectory forecasts.
3.  **Evaluation:** All models are evaluated on the test set using RMSE, MAPE, ACC (Directional Accuracy), and R2 metrics. Results are printed to the console.
4.  **Visualization & Logging:**
    -   Prediction plots for rolling and trajectory forecasts are saved to `results/plots/`.
    -   LSTM training loss curves (for Pure Bi-LSTM, Hybrid LSTMs, and ICEEMDAN-Seq2Seq LSTM) are saved and logged to TensorBoard.
    -   TensorBoard logs also include model graphs and histograms.
5.  **Artifact Saving:**
    -   Trained ICEEMDAN-Seq2Seq LSTM models (full SavedModel and checkpoints) are saved to versioned directories in `models/saved_weights/ICEEMDAN_Seq2Seq_LSTM_versions/`.
    -   Data scalers (for IMFs and prices) used during ICEEMDAN processing are saved to `models/saved_scalers/`.

**Configuration Highlights (see `src/config.py` for details):**
- `FORCE_CPU_TRAINING`: Force TensorFlow to use CPU.
- `HYBRID_USE_LOG_TRANSFORM`: Determines if Hybrid models use original or log-transformed prices.
- `ICEEMDAN_DATA_FORCE_REPROCESS`: Force reprocessing of ICEEMDAN data.
- `ICEEMDAN_LOAD_SAVED_MODEL`, `ICEEMDAN_FORCE_RETRAIN`: Control loading/training of ICEEMDAN-Seq2Seq model.

## TensorBoard Integration

TensorBoard is integrated with the LSTM training process (Pure Bi-LSTM, LSTM components of Hybrid models, and the new ICEEMDAN-Seq2Seq LSTM model) to provide detailed visualizations.

**To use TensorBoard:**

1.  While the main script is running or after it completes, open a **new terminal**.
2.  Navigate to the `nvda_stock_predictor` directory (i.e., `cd path/to/your/NVDA_Project/nvda_stock_predictor`).
3.  Run the TensorBoard command:
    ```bash
    tensorboard --logdir logs/tensorboard
    ```
4.  Open the URL provided by TensorBoard (usually `http://localhost:6006/`) in your web browser.

You can then explore:
*   **Scalars:** Training/validation loss for each LSTM model.
*   **Graphs:** The architecture of the Keras models.
*   **Distributions & Histograms:** Weight and activation distributions over training epochs.

Logs are organized by model name (e.g., `Pure_BiLSTM`, `Hybrid_ARIMA(1,1,1)+LSTM_Residual_Original Scale`, `ICEEMDAN-Seq2Seq LSTM_YYYYMMDD-HHMMSS`) and timestamp within the `logs/tensorboard/` directory (relative to project root).

## Models Compared

The project implements and compares the following models:
- Naive Forecast (Rolling & Trajectory)
- ARIMA(1,1,1)
- Auto ARIMA
- Pure Bi-LSTM (Bidirectional LSTM)
- Hybrid ARIMA(1,1,1) + LSTM
- Hybrid Auto ARIMA + LSTM
- ICEEMDAN-Seq2Seq LSTM with Bahdanau Attention

## Results

Model evaluation metrics (RMSE, MAPE, ACC, R2) are printed to the console. Prediction plots and LSTM loss curves are saved in `results/plots/`. TensorBoard provides detailed LSTM training insights. Saved ICEEMDAN-Seq2Seq models and scalers facilitate re-runs and inference.

*(Add more details about specific findings or conclusions here)*

## Documentation

The `docs/` folder contains supplementary documents. For the most up-to-date understanding of the project's methodology and implementation, please refer to the source code in the `src/` directory and this README file.
<!-- - `NVDA預測 Hybrid (ARIMA + LSTM) 與多模型比較 (1) 1df5add553d58013bd3ad0f6562446e6.md` -->

## Contributing

*(Add contribution guidelines if applicable)*

## License

*(Specify project license if applicable)*