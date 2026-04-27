from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainConfig(BaseSettings):
    """Training configuration.

    All fields can be overridden via environment variables e.g. ``NUM_EPOCHS=50``.
    For list fields use JSON syntax: ``LAGS='[1,2,3,6,12,24]'``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_parse_none_str="null",
    )

    # --- model ---
    model_type: Literal["simple", "quantile"] = "quantile"
    forecast_horizon: int = Field(12, gt=0)
    target_col: str = "no2"

    # --- optimiser ---
    num_epochs: int = Field(10, gt=0)
    lr: float = Field(1e-3, gt=0, lt=1)

    # --- features ---
    temporal_feature_cols: list[str] = [
        "day_of_week",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]
    measurements_feature_cols: list[str] = ["temp", "feuchte", "windge", "no2"]
    spatial_feature_cols: list[str] = ["station_code"]
    lags: list[int] = [1, 2, 3, 6, 12, 24]

    # --- data ---
    start: datetime = datetime(2026, 1, 1)
    end: datetime = datetime(2026, 3, 31)
    retrieve_new_measurements: bool = True
    select_features: bool = False

    # --- derived (not overridable directly) ---
    @computed_field  # type: ignore[prop-decorator]
    @property
    def lag_feature_cols(self) -> list[str]:
        return [
            f"{col}_lag{lag}"
            for lag in self.lags
            for col in self.measurements_feature_cols
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def feature_cols(self) -> list[str]:
        return (
            self.temporal_feature_cols
            + self.spatial_feature_cols
            + self.lag_feature_cols
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def target_cols(self) -> list[str]:
        return [
            f"target_{self.target_col}_lag{i + 1}"
            for i in range(self.forecast_horizon)
        ]
