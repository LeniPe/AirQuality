import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from src.dataset import TabularTimeSeriesDataset
from torch.utils.data import DataLoader
import torch


def validate(feature_cols, target_cols, model):
    test_df = pd.read_csv("data/processed/test.csv")

    test_df.dropna(inplace=True)

    mse_p1 = mean_squared_error(
        test_df[target_cols], test_df[["no2_lag1"] * len(target_cols)]
    )

    test_dataset = TabularTimeSeriesDataset(
        path="data/processed/test.csv",
        feature_cols=feature_cols,
        target_cols=target_cols,
        station_id="0104",
    )

    test_loader = DataLoader(test_dataset, batch_size=256, num_workers=2)
    model.eval()
    y_preds = []
    y_trues = []
    with torch.no_grad():
        for X, station_code, y in test_loader:
            X, station_code, y = X.to("cpu"), station_code.to("cpu"), y.to("cpu")
            if y.ndim == 1:
                y = y.unsqueeze(1)
            pred = model(X, station_code)
            y_preds.append(pred.cpu().numpy())
            y_trues.append(y.cpu().numpy())

    y_preds = np.concatenate(y_preds, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)
    print(f"y_trues shape: {y_trues.shape}, y_preds shape: {y_preds.shape}")

    mse_per_horizon = np.mean((y_trues - y_preds) ** 2, axis=0)
    print(f"MSE per forecast horizon: {mse_per_horizon}")

    mse_model = mean_squared_error(y_trues, y_preds)

    print(f"Persistence 1h MSE: {mse_p1:.4f}\nModel MSE: {mse_model:.4f}")
