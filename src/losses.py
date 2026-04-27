from collections.abc import Sequence

import torch
import torch.nn as nn


class PinballLoss(nn.Module):
    def __init__(self, quantiles: Sequence[float]):
        super().__init__()

        if len(quantiles) == 0:
            raise ValueError("quantiles must contain at least one value")

        values = torch.tensor([float(q) for q in quantiles], dtype=torch.float32)
        if torch.any(values <= 0.0) or torch.any(values >= 1.0):
            raise ValueError("all quantiles must be in the open interval (0, 1)")

        self.register_buffer("quantiles", values)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if preds.ndim != 3:
            raise ValueError(
                "preds must have shape [batch_size, forecast_horizon, num_quantiles]"
            )

        if target.ndim == 1:
            target = target.unsqueeze(1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)
        if target.ndim != 3:
            raise ValueError(
                "target must have shape [batch_size, forecast_horizon] or [batch_size, forecast_horizon, 1]"
            )

        if preds.shape[0] != target.shape[0] or preds.shape[1] != target.shape[1]:
            raise ValueError(
                "preds and target must match in batch_size and forecast_horizon dimensions"
            )

        if preds.shape[2] != self.quantiles.numel():
            raise ValueError(
                "num_quantiles in preds must match the quantiles configured in PinballLoss"
            )

        quantiles = self.quantiles.view(1, 1, -1).to(device=preds.device, dtype=preds.dtype)
        errors = target - preds

        # Pinball loss: max(q * error, (q - 1) * error)
        loss = torch.maximum(quantiles * errors, (quantiles - 1.0) * errors)
        return loss.mean()
