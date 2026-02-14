from src.dataset import TabularTimeSeriesDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def predict(feature_cols, target_col, model, writer: SummaryWriter):

    test_dataset = TabularTimeSeriesDataset(
        path="data/processed/test.csv",
        feature_cols=feature_cols,
        target_col=target_col,
        station_id="0104",
    )

    test_loader = DataLoader(test_dataset, batch_size=256, num_workers=2)
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to("cpu"), y.to("cpu")
            if y.ndim == 1:
                y = y.unsqueeze(1)
            pred = model(X)
            plt.plot(pred.cpu().numpy(), label="Predicted")
            plt.plot(y.cpu().numpy(), label="Observed")
            plt.legend()

            writer.add_figure("Predictions vs Observed", plt.gcf())
            plt.savefig("output/predictions.png")
            break
