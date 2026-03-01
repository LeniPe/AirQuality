import torch
from torch.utils.data import Dataset
import pandas as pd


class TabularTimeSeriesDataset(Dataset):
    def __init__(
        self, path, feature_cols: list[str], target_cols: list[str], station_id=None
    ):
        """
        path: str
            Path to processed data file (parquet or csv)
        feature_cols: list[str]
            Column names used as input features
        target_cols: list[str]
            Column names of the target variables
        """
        self.df = pd.read_csv(path, dtype={"station_id": str})
        if station_id is not None:
            self.df = self.df[self.df["station_id"] == station_id]

        self.df.reset_index(drop=True, inplace=True)

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
