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
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            if y.ndim == 1:
                y = y.unsqueeze(1)
            pred = model(X)
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

    model = SimpleRegressor(num_features=len(feature_cols)).to("cpu")
    writer.add_graph(model, torch.randn(1, len(feature_cols)).to("cpu"))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0.0

        for X, y in train_loader:
            X, y = X.to("cpu"), y.to("cpu")
            if y.ndim == 1:
                y = y.unsqueeze(1)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch + 1)
        test(test_loader, model, criterion)

    writer.flush()
    return model
