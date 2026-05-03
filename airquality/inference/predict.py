import json
import datetime
from pathlib import Path
import numpy as np
import joblib
import pandas as pd

import torch
from sklearn.preprocessing import StandardScaler

from airquality.models.regressor import SimpleRegressor, QuantileRegressor
from airquality.data.fetch import fetch_hourly_measurements_on_the_fly
from airquality.data.preprocessing import preprocess_inference_measurements, map_param_name_to_id
from airquality.data.time_utils import to_local_datetime, from_local_timestamp


DEVICE = torch.device("cpu")
CHECKPOINT_PATH = Path("output/model_checkpoint.pth")
STATION_MAP_PATH = Path("project_data/station_mapping.json")
OUTPUT_PATH = Path("output/predictions.png")


def predict_series(
    feature_cols: list[str],
    target_col: str,
    lags: list[int],
    model: SimpleRegressor,
    station_id: str,
    requested_dt: datetime.datetime,
    parameter: str = "NO2",
) -> tuple[list[datetime.datetime], np.ndarray]:
    parameter_name = parameter.strip().lower()
    if parameter_name != target_col.lower():
        raise ValueError(
            f"Unsupported parameter '{parameter}'. This model supports '{target_col.upper()}' only."
        )

    requested_dt = to_local_datetime(requested_dt)
    history_hours = max(lags) + 1
    scaler: StandardScaler = joblib.load("project_data/std_scaler.joblib")
    param_names = list(scaler.feature_names_in_)  # type: ignore[attr-defined]
    param_ids = map_param_name_to_id(param_names)

    raw_measurements = fetch_hourly_measurements_on_the_fly(
        station_id=station_id,
        start=requested_dt - datetime.timedelta(hours=history_hours),
        end=requested_dt,
        param_ids=param_ids,
    )

    latest_observation_time = from_local_timestamp(max(raw_measurements.timestamp))

    if latest_observation_time < requested_dt:
        print(f"Latest observation time {latest_observation_time} is before the requested datetime {requested_dt}. Adjusting requested datetime to latest observation time.")
        requested_dt = latest_observation_time
        raw_measurements = fetch_hourly_measurements_on_the_fly(
            station_id=station_id,
            start=requested_dt - datetime.timedelta(hours=history_hours),
            end=requested_dt,
            param_ids=param_ids,
        )

    df = preprocess_inference_measurements(
        param_names=param_names,
        station_id=station_id,
        lags=lags,
        start=requested_dt - datetime.timedelta(hours=history_hours),
        end=requested_dt,
        measurements_df=raw_measurements,
    )
    if df.empty:
        raise ValueError("No processed inference data available.")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(
        requested_dt.tzinfo
    )

    df = df[(df["station_id"] == station_id) & (df["datetime"] == requested_dt)]
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise ValueError(
            f"No data found for station_id='{station_id}' at '{requested_dt.isoformat()}'"
        )

    model.eval()
    row = df.iloc[0]
    X = (
        torch.tensor(row[feature_cols].values.astype(np.float32), dtype=torch.float32)
        .unsqueeze(0)
        .to(DEVICE)
    )
    station_code = torch.tensor([int(row["station_code"])], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        pred = model(X, station_code)

    forecast_horizon = pred.shape[1]
    pred_times = [
        requested_dt + datetime.timedelta(hours=i + 1)
        for i in range(forecast_horizon)
    ]

    pred = inverse_scale_target(pred, target_col=target_col)
    pred_values = pred[0].detach().cpu().numpy().astype(float)

    return pred_times, pred_values


def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    with open(STATION_MAP_PATH, "r") as f:
        station_mapping = json.load(f)

    config = checkpoint["config"]
    forecast_horizon = config["forecast_horizon"]
    target_col = config["target_col"]
    model_type = config.get("model_type", "simple")

    target_cols = [f"target_{target_col}_lag{i + 1}" for i in range(forecast_horizon)]

    if model_type == "simple":
        model = SimpleRegressor(
            num_features=len(config["feature_cols"]),
            forecast_horizon=forecast_horizon,
            num_stations=len(station_mapping),
        )
    elif model_type == "quantile":
        model = QuantileRegressor(
            num_features=len(config["feature_cols"]),
            forecast_horizon=forecast_horizon,
            num_stations=len(station_mapping),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    return (
        model,
        config["feature_cols"],
        target_cols,
        config["target_col"],
        config["lags"],
        model_type,
    )


def inverse_scale_target(x, target_col: str):
    scaler: StandardScaler = joblib.load("project_data/std_scaler.joblib")
    index = list(scaler.feature_names_in_).index(target_col)  # type: ignore[attr-defined]
    x_original = x * scaler.scale_[index] + scaler.mean_[index]
    return x_original