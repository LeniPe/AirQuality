import json
from typing import Literal, Optional
import pandas as pd
from airquality.data.dataset import TabularTimeSeriesDataset
from torch.utils.data import DataLoader
import torch
from airquality.models.regressor import SimpleRegressor, QuantileRegressor
from airquality.models.losses import PinballLoss
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def validate(feature_cols, target_cols, model, loss_fn, device="cpu"):
    df = pd.read_csv("project_data/processed/val.csv", dtype={"station_id": str}).reset_index(
        drop=True
    )
    val_dataset = TabularTimeSeriesDataset(
        df,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    dataloader = DataLoader(val_dataset, batch_size=256, num_workers=2)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, station_code, y in dataloader:
            X, station_code, y = X.to(device), station_code.to(device), y.to(device)
            if y.ndim == 1:
                y = y.unsqueeze(1)
            pred = model(X, station_code)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    return test_loss


def feature_selection(feature_cols, target_cols, num_epochs=10):
    selected_features = [
        "station_code"
    ]  # Always include station_code as a spatial feature
    available_features = feature_cols.copy()
    available_features.remove("station_code")
    for j in range(5):  # Select top 5 features
        loss = float("inf")
        for i in available_features:
            model = train_model(
                feature_cols=selected_features + [i],
                target_cols=target_cols,
                num_epochs=num_epochs,
            )
            new_loss = validate(
                feature_cols=selected_features + [i],
                target_cols=target_cols,
                model=model,
                loss_fn=nn.MSELoss(),
                device="cpu",
            )
            print(f"   Validation loss with feature {i}: {new_loss:.4f}")
            if new_loss < loss:
                selected_feature = i
                loss = new_loss
        print(
            f"Feature number {j + 1}: {selected_feature} with validation loss: {loss:.4f}"
        )
        selected_features.append(selected_feature)
        available_features.remove(selected_feature)
    return selected_features


def train_model(
    feature_cols,
    target_cols,
    num_epochs: int = 20,
    writer: Optional[SummaryWriter] = None,
    model_type: Literal["simple", "quantile"] = "simple",
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    batch_size: int = 256,
    device = "cpu",
    lr: float = 1e-3,
) -> tuple[torch.nn.Module, float]:
    df = pd.read_csv("project_data/processed/train.csv", dtype={"station_id": str}).reset_index(
        drop=True
    )
    train_dataset = TabularTimeSeriesDataset(
        df,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    df_val = pd.read_csv("project_data/processed/val.csv", dtype={"station_id": str}).reset_index(
        drop=True
    )
    val_dataset = TabularTimeSeriesDataset(
        df_val,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)


    num_stations = json.load(open("project_data/station_mapping.json", "r"))
    if model_type == "simple":
        model = SimpleRegressor(
            num_features=len(feature_cols),
            num_stations=len(num_stations),
            embedding_dim=8,
            forecast_horizon=len(target_cols),
        ).to(device)
        criterion = nn.MSELoss()
    elif model_type == "quantile":
        model = QuantileRegressor(
            num_features=len(feature_cols),
            num_stations=len(num_stations),
            embedding_dim=8,
            forecast_horizon=len(target_cols),
            quantiles=quantiles,
        ).to(device)
        criterion = PinballLoss(quantiles=quantiles)
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Use 'simple' or 'quantile'."
        )

    if writer is not None:
        dummy_x = torch.randn(1, len(feature_cols)).to(device)
        dummy_station = torch.zeros(1, dtype=torch.long).to(device)
        writer.add_graph(model, (dummy_x, dummy_station))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for X, station_code, y in train_loader:
            X, station_code, y = X.to(device), station_code.to(device), y.to(device)
            if y.ndim == 1:
                y = y.unsqueeze(1)
            optimizer.zero_grad()
            preds = model(X, station_code)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if writer is not None:
            writer.add_scalar("Loss/train", avg_loss, epoch + 1)

        # Validation
        if writer is None:
            continue
        model.eval()
        all_q_lower = []
        all_q_mid = []
        all_q_upper = []
        all_y = []
        with torch.no_grad():
            for X, station_code, y in val_loader:
                X, station_code, y = X.to(device), station_code.to(device), y.to(device)
                if y.ndim == 1:
                    y = y.unsqueeze(1)
                pred = model(X, station_code)
                all_q_lower.append(pred[:, :, 0])
                all_q_mid.append(pred[:, :, 1])
                all_q_upper.append(pred[:, :, 2])
                all_y.append(y)
        q_lower = torch.cat(all_q_lower, dim=0)
        q_mid = torch.cat(all_q_mid, dim=0)
        q_upper = torch.cat(all_q_upper, dim=0)
        y   = torch.cat(all_y, dim=0)

        interval_cov = (
            (y >= q_lower) &
            (y <= q_upper)
        ).float().mean()
        interval_width = (q_upper - q_lower).mean()
        val_loss = criterion(torch.stack([q_lower, q_mid, q_upper], dim=-1), y)

        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("val/interval_coverage", interval_cov, epoch + 1)
        writer.add_scalar("val/interval_width", interval_width, epoch + 1)


    if writer is not None:
        writer.flush()
    return model, val_loss
