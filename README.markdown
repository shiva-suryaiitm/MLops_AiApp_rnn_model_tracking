# Stock Prediction Training Pipeline

## Overview
This proprietary folder trains LSTM (prices) and GRU (volumes) models using MLflow for logging and grid search, and DVC for data/model versioning. It exports MongoDB data, trains models, and saves the best weights based on validation error.

## Prerequisites
- Python 3.8+
- DVC (`pip install dvc`)
- MLflow (`pip install mlflow`)
- MongoDB with `stock_prices` and `stock_volumes`
- NVIDIA GPU (optional, for faster training)

## Setup
1. **Install dependencies**:
   ```bash
   pip install dvc mlflow pymongo torch numpy scikit-learn pyyaml
   ```

2. **Prepare MongoDB**:
   - Populate `portfolio_management`:
     - `stock_prices`: `{company: str, stock_price: float, date: datetime}`
     - `stock_volumes`: `{company: str, volume: int, date: datetime}`
   - Run:
     ```bash
     python3 generate_sample_data.py
     ```

3. **Run DVC Pipeline**:
   ```bash
   mkdir -p data models
   dvc init
   dvc repro
   dvc push
   ```
   - Exports data to `data/`.
   - Trains models, saves weights to `models/`.

4. **View MLflow Logs**:
   ```bash
   mlflow ui
   ```
   Access: `http://localhost:5000`

5. **Copy Weights for Inference**:
   ```bash
   cp models/price_model.pth ../inference/models/
   cp models/volume_model.pth ../inference/models/
   ```

## Configuration
- **MongoDB**: `mongodb://localhost:27017/`. Update `export_data.py` if different.
- **Hyperparameters**: Defined in `hyperparameters.yaml`.
- **MLflow**: Logs to `mlruns/`.

## Project Structure
```
training/
├── train_models.py           # Training with MLflow
├── export_data.py            # Data export
├── generate_sample_data.py   # Sample data
├── hyperparameters.yaml      # Grid search parameters
├── dvc.yaml                  # DVC pipeline
├── data/                     # Versioned data
├── models/                   # Versioned weights
└── README.md                 # Documentation
```

## Usage
- **DVC Pipeline**:
  - Export: `python3 export_data.py`
  - Train: `python3 train_models.py`
  - Run: `dvc repro`

## Troubleshooting
- **MongoDB**: Verify connectivity with `mongosh`.
- **Data**: Ensure 200+ days in MongoDB.
- **MLflow**: Check `mlflow ui`, ensure `mlruns/`.
- **DVC**: Check `dvc status`, `dvc repro`.

## License
Proprietary