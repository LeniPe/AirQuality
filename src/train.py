import json
from src.dataset import TabularTimeSeriesDataset
from torch.utils.data import DataLoader
import torch
from src.model import SimpleRegressor
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def test(dataloader, model, loss_fn, device="cpu"):
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
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")


def train_model(feature_cols, target_col, writer: SummaryWriter, num_epochs: int = 20):

    train_dataset = TabularTimeSeriesDataset(
        path="data/processed/train.csv",
        feature_cols=feature_cols,
        target_col=target_col,
    )

    test_dataset = TabularTimeSeriesDataset(
        path="data/processed/test.csv", feature_cols=feature_cols, target_col=target_col
    )

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=2
    )

    test_loader = DataLoader(test_dataset, batch_size=256, num_workers=2)

    num_stations = json.load(open("data/station_mapping.json", "r"))
    model = SimpleRegressor(
        num_features=len(feature_cols), num_stations=len(num_stations), embedding_dim=8
    ).to("cpu")
    
    dummy_x = torch.randn(1, len(feature_cols)).to("cpu")
    dummy_station = torch.zeros(1, dtype=torch.long).to("cpu")

    writer.add_graph(model, (dummy_x, dummy_station))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0.0

        for X, station_code, y in train_loader:
            X, station_code, y = X.to("cpu"), station_code.to("cpu"), y.to("cpu")
            if y.ndim == 1:
                y = y.unsqueeze(1)
            optimizer.zero_grad()
            preds = model(X, station_code)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch + 1)
        test(test_loader, model, criterion)

    writer.flush()
    return model
