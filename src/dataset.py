import torch
from torch.utils.data import Dataset
import pandas as pd


class TabularTimeSeriesDataset(Dataset):
    def __init__(self, path, feature_cols, target_col, station_id=None):
        """
        path: str
            Path to processed data file (parquet or csv)
        feature_cols: list[str]
            Column names used as input features
        target_col: str
            Column name of the target variable
        """
        self.df = pd.read_csv(path, dtype={"station_id": str})
        if not station_id is None:
            self.df = self.df[self.df["station_id"] == station_id]

        self.feature_cols = feature_cols
        self.target_col = target_col

        self.X = torch.tensor(self.df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(self.df[target_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
