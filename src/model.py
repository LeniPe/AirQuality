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


class QuantileRegressor(nn.Module):
    def __init__(
        self,
        num_features,
        num_stations,
        embedding_dim=8,
        forecast_horizon=1,
        quantiles=(0.1, 0.5, 0.9),
    ):
        super().__init__()

        quantiles = tuple(float(q) for q in quantiles)
        if len(quantiles) == 0:
            raise ValueError("quantiles must contain at least one value")
        if any(q <= 0.0 or q >= 1.0 for q in quantiles):
            raise ValueError("all quantiles must be in the open interval (0, 1)")
        if list(quantiles) != sorted(quantiles):
            raise ValueError("quantiles must be sorted in ascending order")

        self.forecast_horizon = forecast_horizon
        self.num_quantiles = len(quantiles)
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

        self.station_embedding = nn.Embedding(
            num_embeddings=num_stations, embedding_dim=embedding_dim
        )
        self.backbone = nn.Sequential(
            nn.Linear(num_features + embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.output = nn.Linear(32, forecast_horizon * self.num_quantiles)

    def forward(self, x, station_code):
        embedded_station = self.station_embedding(station_code)
        x = torch.cat([x, embedded_station], dim=1)
        latent = self.backbone(x)
        pred = self.output(latent)

        # Return shape: [batch_size, forecast_horizon, num_quantiles]
        return pred.view(-1, self.forecast_horizon, self.num_quantiles)
