import json
import datetime
from pathlib import Path
import numpy as np
import joblib

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from dataset import TabularTimeSeriesDataset
from model import SimpleRegressor


DEVICE = torch.device("cpu")
TEST_PATH = Path("data/processed/test.csv")
CHECKPOINT_PATH = Path("output/model_checkpoint.pth")
STATION_MAP_PATH = Path("data/station_mapping.json")
OUTPUT_PATH = Path("output/predictions.png")


def predict(
    feature_cols,
    target_cols,
    model,
    station_id="0104",
    datetime_str="2025-11-08 11:00:00",
):
    dataset = TabularTimeSeriesDataset(
        path=str(TEST_PATH),
        feature_cols=feature_cols,
        target_cols=target_cols,
        station_id=station_id,
    )
    print(dataset.df.datetime.min(), dataset.df.datetime.max())
    # Filter dataset for specific date
    target_date = datetime.datetime.timestamp(
        datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    )
    print(target_date)
    dataset.df = dataset.df[dataset.df["timestamp"] == target_date]

    if dataset.df.empty:
        print(f"No data found for {datetime_str}")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()

    pred_time = dataset.df["timestamp"].iloc[0]

    with torch.no_grad():
        for X, station_code, y in loader:
            X = X.to(DEVICE)
            station_code = station_code.to(DEVICE)
            y = y.to(DEVICE)

            if y.ndim == 1:
                y = y.unsqueeze(1)

            pred = model(X, station_code)

            forecast_horizon = pred.shape[1]
            times = [
                datetime.datetime.fromtimestamp(pred_time + (i) * 3600)
                for i in range(forecast_horizon + 1)
            ]

            pred = load_scaler(pred)
            y = load_scaler(y)
            X = load_scaler(X)
            plot_predictions(times, pred[0].numpy(), y[0].numpy(), now=X[0][9].numpy())
            break


def plot_predictions(times, pred, target, now):
    pred = np.insert(pred, 0, now)
    target = np.insert(target, 0, now)
    plt.figure()
    plt.title(f"NO2 concentration as of {times[0]}")
    plt.plot(times, pred, label="Predicted")
    plt.plot(times, target, label="Observed")
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter("%Y-%m-%d %H:%M")
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    plt.close()


def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    with open(STATION_MAP_PATH, "r") as f:
        station_mapping = json.load(f)

    config = checkpoint["config"]
    forecast_horizon = config["forecast_horizon"]
    target_col = config["target_col"]

    target_cols = [f"target_{target_col}_lag{i + 1}" for i in range(forecast_horizon)]

    model = SimpleRegressor(
        num_features=len(config["feature_cols"]),
        forecast_horizon=forecast_horizon,
        num_stations=len(station_mapping),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    return model, config["feature_cols"], target_cols


def load_scaler(x, index=3):
    scaler: StandardScaler = joblib.load("data/std_scaler.joblib")
    # scaler.inverse_transform
    print(scaler.feature_names_in_[index])
    x_original = x * scaler.scale_[index] + scaler.mean_[index]
    return x_original


if __name__ == "__main__":
    model, feature_cols, target_cols = load_model()
    print(feature_cols)
    predict(feature_cols, target_cols, model)
