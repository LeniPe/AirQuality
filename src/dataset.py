import torch
from torch.utils.data import Dataset
import pandas as pd


class TabularTimeSeriesDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, feature_cols: list[str], target_cols: list[str]
    ):
        self.df = df
        self.feature_cols = feature_cols
        self.target_cols = target_cols

        self.X = torch.tensor(self.df[feature_cols].values, dtype=torch.float32)
        self.station_code = torch.tensor(
            self.df["station_code"].values, dtype=torch.long
        )
        self.y = torch.tensor(self.df[target_cols].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.station_code[idx], self.y[idx]
