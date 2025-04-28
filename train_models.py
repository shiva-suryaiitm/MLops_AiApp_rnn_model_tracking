import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import logging
import mlflow
import mlflow.pytorch
import yaml
from itertools import product
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configuration
load_dotenv(dotenv_path= os.path.dirname(__file__) + '/train.env.env')

SEQUENCE_LENGTH = int(os.getenv('SEQUENCE_LENGTH'), 100)
VALIDATION_DAYS = int(os.getenv('VALIDATION_DAYS'), 30)
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', "http://localhost:5000")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.3):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

def preprocess_data(data, field, sequence_length=SEQUENCE_LENGTH):
    """Preprocess data."""
    values = np.array([item[field] for item in data]).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)
    sequences = []
    for i in range(len(scaled_values) - sequence_length):
        sequences.append(scaled_values[i:i + sequence_length])
    return np.array(sequences), scaler

def train_model(model, X_train, y_train, X_val, y_val, params):
    """Train and log to MLflow."""
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.FloatTensor(y_train).to(DEVICE)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

    X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
    y_val_tensor = torch.FloatTensor(y_val).to(DEVICE)

    for epoch in tqdm(range(params["epochs"]), desc = "Training", unit="epoch"):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

    return val_loss

def grid_search(model_class, X_train, y_train, X_val, y_val, hyperparams, model_name):
    """Perform grid search."""
    best_val_loss = float("inf")
    best_params = None
    best_model = None

    param_grid = list(product(*[hyperparams[key] for key in sorted(hyperparams.keys())]))
    for param_set in param_grid:
        params = {
            "hidden_size": param_set[0],
            "dropout": param_set[1],
            "epochs": param_set[2],
            "learning_rate": param_set[3]
        }
        params["batch_size"] = 32 if model_name == "LSTM" else 64

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            model = model_class(hidden_size=params["hidden_size"], dropout=params["dropout"]).to(DEVICE)
            val_loss = train_model(model, X_train, y_train, X_val, y_val, params)
            mlflow.log_metric("final_val_loss", val_loss)
            mlflow.pytorch.log_model(model, f"{model_name}_model")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                best_model = model

    return best_model, best_params, best_val_loss

def main():
    """Train models."""
    # Load data
    with open("data/prices.json", "r") as f:
        price_data = json.load(f)
    with open("data/volumes.json", "r") as f:
        volume_data = json.load(f)

    # Split data
    validation_start = datetime.now() - timedelta(days=VALIDATION_DAYS)
    price_train = [d for d in price_data if datetime.strptime(d["date"], "%Y-%m-%dT%H:%M:%S") < validation_start]
    price_val = [d for d in price_data if datetime.strptime(d["date"], "%Y-%m-%dT%H:%M:%S") >= validation_start]
    volume_train = [d for d in volume_data if datetime.strptime(d["date"], "%Y-%m-%dT%H:%M:%S") < validation_start]
    volume_val = [d for d in volume_data if datetime.strptime(d["date"], "%Y-%m-%dT%H:%M:%S") >= validation_start]

    # Preprocess data
    X_price_train, _ = preprocess_data(price_train, "stock_price")
    y_price_train = X_price_train[:, -1, :]
    X_price_train = X_price_train[:, :-1, :]
    X_price_val, _ = preprocess_data(price_val, "stock_price")
    y_price_val = X_price_val[:, -1, :]
    X_price_val = X_price_val[:, :-1, :]

    X_volume_train, _ = preprocess_data(volume_train, "volume")
    y_volume_train = X_volume_train[:, -1, :]
    X_volume_train = X_volume_train[:, :-1, :]
    X_volume_val, _ = preprocess_data(volume_val, "volume")
    y_volume_val = X_volume_val[:, -1, :]
    X_volume_val = X_volume_val[:, :-1, :]

    # Load hyperparameters
    with open("hyperparameters.yaml", "r") as f:
        hyperparams = yaml.safe_load(f)

    # Train price model

    # using mlflow to log the model
    # set the experiment name
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("stock_price_prediction")

    with mlflow.start_run():
        best_price_model, best_price_params, best_price_val_loss = grid_search(
            LSTMModel, X_price_train, y_price_train, X_price_val, y_price_val,
            hyperparams["lstm"], "LSTM"
        )
        mlflow.log_params(best_price_params)
        mlflow.log_metric("best_val_loss", best_price_val_loss)
        torch.save(best_price_model.state_dict(), os.path.dirname(__file__) + "/models/price_model.pth")
        mlflow.pytorch.log_model(best_price_model, "best_price_model")
        logger.info("Saved best price model weights")

    # Train volume model
    mlflow.set_experiment("volume_prediction")
    with mlflow.start_run():
        best_volume_model, best_volume_params, best_volume_val_loss = grid_search(
            GRUModel, X_volume_train, y_volume_train, X_volume_val, y_volume_val,
            hyperparams["gru"], "GRU"
        )
        mlflow.log_params(best_volume_params)
        mlflow.log_metric("best_val_loss", best_volume_val_loss)
        torch.save(best_volume_model.state_dict(), os.path.dirname(__file__) + "models/volume_model.pth")
        mlflow.pytorch.log_model(best_volume_model, "best_volume_model")
        logger.info("Saved best volume model weights")

if __name__ == "__main__":
    main()