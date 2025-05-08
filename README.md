# NVDA Stock Price Prediction Project

This project aims to predict NVIDIA (NVDA) stock prices using various time series forecasting models, including ARIMA, LSTM, and Hybrid ARIMA+LSTM approaches.

## Project Structure

```
nvda_stock_predictor/
├── data/
│   ├── raw/                  # Raw data (e.g., NVDA_stock_data_new.csv)
│   └── processed/            # (Optional) Processed data
├── docs/                     # Documentation (e.g., project description MD file)
├── notebooks/                # (Optional) Jupyter notebooks for exploration
├── results/
│   ├── plots/                # Saved plots (loss curves, predictions)
│   └── metrics/              # (Optional) Saved metrics files
├── scripts/                  # (Optional) Utility scripts
├── src/                      # Source code package
│   ├── __init__.py           # Makes src a package
│   ├── config.py             # Configuration constants and paths
│   ├── data_processing.py    # Data loading and splitting functions
│   ├── evaluation.py         # Model evaluation metric functions
│   ├── main.py               # Main execution script orchestrating the pipeline
│   ├── visualization.py      # Plotting functions
│   └── models/               # Model implementations sub-package
│       ├── __init__.py       # Makes models a sub-package
│       ├── arima.py
│       ├── hybrid.py
│       ├── lstm.py
│       └── naive.py
├── logs/                     # (Optional) Log files
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
    **Note on GPU Support:** The `requirements.txt` file specifies `tensorflow-gpu`. To use GPU acceleration for LSTM training, you must have a compatible NVIDIA GPU, the correct NVIDIA driver, CUDA Toolkit, and cuDNN library installed on your system. Please refer to the official TensorFlow documentation for the specific CUDA/cuDNN versions required for TensorFlow 2.10. If you do not have a compatible GPU or do not wish to install the GPU dependencies, you can modify `requirements.txt` to use `tensorflow-cpu` instead.

## Data

The raw stock data is expected in the `data/raw/` directory. The primary data file used is `NVDA_stock_data_new.csv`.

## Usage

**Important:** Due to the use of relative imports within the `src` package, the main script should be run as a module from the **project's root directory** (i.e., the directory that contains the `nvda_stock_predictor` package directory).

1.  Navigate to the project's root directory in your terminal.
2.  Run the following command:

    ```bash
    python -m nvda_stock_predictor.src.main
    ```

This command tells Python to execute the `main` module located inside the `nvda_stock_predictor.src` package.

The script will:
- Load and preprocess the data from `data/raw/`.
- Split the data into training, validation, and test sets.
- Train Naive, ARIMA(1,1,1), Auto ARIMA, Pure LSTM, and Hybrid models.
- Perform rolling and trajectory forecasts on the test set.
- Evaluate models using RMSE, MAPE, ACC, and R2 metrics.
- Save prediction plots to `results/plots/`.
- Print evaluation metrics to the console.

## Models Compared

- Naive Forecast
- ARIMA(1,1,1)
- Auto ARIMA
- Pure LSTM
- Hybrid ARIMA(1,1,1) + LSTM
- Hybrid Auto ARIMA + LSTM

## Results

Model evaluation metrics are printed to the console upon script completion. Plots comparing actual vs. predicted values for rolling and trajectory forecasts are saved in the `results/plots/` directory.

*(Add more details about specific findings or conclusions here)*

## Documentation

Refer to the document in the `docs/` folder for detailed project requirements and methodology:
- `NVDA預測 Hybrid (ARIMA + LSTM) 與多模型比較 (1) 1df5add553d58013bd3ad0f6562446e6.md`

## Contributing

*(Add contribution guidelines if applicable)*

## License

*(Specify project license if applicable)*