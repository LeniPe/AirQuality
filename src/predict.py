import json
import datetime
from pathlib import Path
import numpy as np
import joblib
import pandas as pd

import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.model import SimpleRegressor, QuantileRegressor
from src.fetch_data import fetch_hourly_measurements_on_the_fly
from src.preprocessing import preprocess_inference_measurements, map_param_name_to_id
from src.time_utils import to_local_datetime, LOCAL_TZ


DEVICE = torch.device("cpu")
CHECKPOINT_PATH = Path("output/model_checkpoint.pth")
STATION_MAP_PATH = Path("data/station_mapping.json")
OUTPUT_PATH = Path("output/predictions.png")


def predict_series(
    feature_cols: list[str],
    target_col: str,
    lags: list[int],
    model: SimpleRegressor,
    station_id: str,
    requested_dt: datetime.datetime,
    parameter: str = "NO2",
) -> tuple[list[datetime.datetime], np.ndarray, list[tuple[datetime.datetime, float]]]:
    parameter_name = parameter.strip().lower()
    if parameter_name != target_col.lower():
        raise ValueError(
            f"Unsupported parameter '{parameter}'. This model supports '{target_col.upper()}' only."
        )

    requested_dt = to_local_datetime(requested_dt)
    history_hours = max(max(lags), 24) + 1
    scaler: StandardScaler = joblib.load("data/std_scaler.joblib")
    param_names = list(scaler.feature_names_in_)  # type: ignore[attr-defined]
    param_ids = map_param_name_to_id(param_names)

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

    observed_points: list[tuple[datetime.datetime, float]] = []
    for lag in sorted(lags, reverse=True):
        lag_feature = f"{target_col}_lag{lag}"
        if lag_feature not in feature_cols:
            continue
        lag_value = inverse_scale_target(
            X[0, feature_cols.index(lag_feature)], target_col=target_col
        )
        observed_points.append(
            (requested_dt - datetime.timedelta(hours=lag), float(lag_value.item()))
        )

    if len(observed_points) == 0:
        raise ValueError(
            f"No observed lag features found for target '{target_col}'. Expected at least one of: "
            + ", ".join([f"{target_col}_lag{lag}" for lag in lags])
        )

    return pred_times, pred_values, observed_points


def predict(
    feature_cols: list[str],
    target_col: str,
    lags: list[int],
    model: SimpleRegressor,
    station_id: str = "0104",
    datetime_str: str = "2026-03-05 08:00:00",
):
    requested_dt = to_local_datetime(
        datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    )
    pred_times, pred_values, observed_points = predict_series(
        feature_cols=feature_cols,
        target_col=target_col,
        lags=lags,
        model=model,
        station_id=station_id,
        requested_dt=requested_dt,
        parameter=target_col,
    )

    observed_times = [t for t, _ in observed_points]
    observed_values = [v for _, v in observed_points]
    plot_predictions(pred_times, pred_values, observed_times, observed_values)


def plot_predictions(pred_times, pred, observed_times, observed_values):
    # Convert all times to naive datetime in local timezone for consistent plotting

    plt.figure()
    plt.title(f"NO2 concentration as of {pred_times[0]}")
    plt.plot(pred_times, pred, label="Predicted", marker="o")
    plt.plot(observed_times, observed_values, label="Observed", marker="o")
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter("%H:%M", tz=LOCAL_TZ)
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    plt.close()


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
    scaler: StandardScaler = joblib.load("data/std_scaler.joblib")
    index = list(scaler.feature_names_in_).index(target_col)  # type: ignore[attr-defined]
    x_original = x * scaler.scale_[index] + scaler.mean_[index]
    return x_original


if __name__ == "__main__":
    model, feature_cols, target_cols, target_col, lags, model_type = load_model()
    print(feature_cols)
    predict(feature_cols, target_col, lags, model)
