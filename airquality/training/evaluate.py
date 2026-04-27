import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from airquality.data.dataset import TabularTimeSeriesDataset
from torch.utils.data import DataLoader
import torch


def _select_point_predictions(pred: torch.Tensor, model) -> torch.Tensor:
    if pred.ndim == 2:
        return pred

    if pred.ndim != 3:
        raise ValueError(
            "Unsupported prediction shape. Expected [batch, horizon] or [batch, horizon, quantiles]."
        )

    # For quantile models, use median quantile as point forecast for MSE metrics.
    quantiles = getattr(model, "quantiles", None)
    if quantiles is None:
        median_idx = pred.shape[2] // 2
        return pred[:, :, median_idx]

    q_tensor = quantiles.detach().cpu().to(dtype=pred.dtype)
    median_idx = int(torch.argmin(torch.abs(q_tensor - 0.5)).item())
    return pred[:, :, median_idx]


def evaluate(feature_cols, target_cols, model):
    test_df = pd.read_csv("project_data/processed/test.csv", dtype={"station_id": str})

    test_df.dropna(inplace=True)
    test_df = test_df[test_df["station_id"] == "0104"]
    test_df.reset_index(drop=True, inplace=True)

    mse_p1 = mean_squared_error(
        test_df[target_cols], test_df[["no2_lag1"] * len(target_cols)]
    )

    test_dataset = TabularTimeSeriesDataset(
        test_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
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
            pred = _select_point_predictions(pred, model)
            y_preds.append(pred.cpu().numpy())
            y_trues.append(y.cpu().numpy())

    y_preds = np.concatenate(y_preds, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)
    print(f"y_trues shape: {y_trues.shape}, y_preds shape: {y_preds.shape}")

    mse_per_horizon = np.mean((y_trues - y_preds) ** 2, axis=0)
    print(f"MSE per forecast horizon: {mse_per_horizon}")

    mse_model = mean_squared_error(y_trues, y_preds)

    print(f"Persistence 1h MSE: {mse_p1:.4f}\nModel MSE: {mse_model:.4f}")