import torch
import torch.nn as nn


class SimpleRegressor(nn.Module):
    def __init__(self, num_features, num_stations, embedding_dim=8, forecast_horizon=1):
        super().__init__()

        self.station_embedding = nn.Embedding(
            num_embeddings=num_stations, embedding_dim=embedding_dim
        )
        self.net = nn.Sequential(
            nn.Linear(num_features + embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_horizon),
        )

    def forward(self, x, station_code):
        embedded_station = self.station_embedding(station_code)
        x = torch.cat([x, embedded_station], dim=1)
        return self.net(x)
