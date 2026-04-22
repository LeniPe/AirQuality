import json
import pandas as pd
from src.fetch_data import (
    fetch_hourly_measurements,
    fetch_parameters,
    fetch_station_details,
    fetch_stations,
)
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import numpy as np
import joblib
from src.time_utils import to_local_timestamp, LOCAL_TZ


def map_param_id_to_name(
    param_ids: list[str], name_type: str = "nameInTable"
) -> list[str]:
    params_df = fetch_parameters()
    if name_type not in ["displayName", "shortName", "nameInTable"]:
        raise ValueError(
            "name_type must be one of 'displayName', 'shortName', 'nameInTable'"
        )
    param_names = []
    for param_id in param_ids:
        param_name = params_df[params_df["paramId"] == int(param_id)][name_type].values[
            0
        ]
        param_names.append(param_name)
    return param_names


def map_param_name_to_id(
    param_names: list[str], name_type: str = "nameInTable"
) -> list[str]:
    params_df = fetch_parameters()
    if name_type not in ["displayName", "shortName", "nameInTable"]:
        raise ValueError(
            "name_type must be one of 'displayName', 'shortName', 'nameInTable'"
        )
    param_ids = (
        params_df.loc[params_df[name_type].isin(param_names), "paramId"]
        .astype(str)
        .tolist()
    )
    return param_ids


def train_test_split(
    df: pd.DataFrame, validation_size: float = 0.1, test_size: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    timestamps = df["datetime"].sort_values().unique()
    val_split_index = int(len(timestamps) * (1 - validation_size - test_size))
    test_split_index = int(len(timestamps) * (1 - test_size))

    train_df = df[df["datetime"].isin(timestamps[:val_split_index])]
    val_df = df[df["datetime"].isin(timestamps[val_split_index:test_split_index])]
    test_df = df[df["datetime"].isin(timestamps[test_split_index:])]
    print(
        f"Train set: {train_df['datetime'].min()} - {train_df['datetime'].max()} with {len(train_df.station_id.unique())} stations"
    )
    print(
        f"Validation set: {val_df['datetime'].min()} - {val_df['datetime'].max()} with {len(val_df.station_id.unique())} stations"
    )
    print(
        f"Test set: {test_df['datetime'].min()} - {test_df['datetime'].max()} with {len(test_df.station_id.unique())} stations"
    )
    return train_df, val_df, test_df


def preprocess_measurements(
    param_names: list[str],
    stations: list[str],
    lags: list[int],
    start: datetime,
    end: datetime,
    target_col: str,
    forecast_horizon: int,
    retrieve_new_measurements: bool = False,
) -> None:

    param_ids = map_param_name_to_id(param_names)

    if retrieve_new_measurements:
        retrieve_measurements(param_ids, stations, start, end)

    measurements_df = prepare_base_measurements_df(
        param_ids=param_ids,
        stations=stations,
        start=start,
        end=end,
        source_dir="data/raw",
    )

    train_df, val_df, test_df = train_test_split(
        measurements_df, validation_size=0.1, test_size=0.1
    )

    station_codes, station_index = pd.factorize(train_df["station_id"])
    station_to_int = {station: i for i, station in enumerate(station_index)}
    with open("data/station_mapping.json", "w") as f:
        json.dump(station_to_int, f)

    train_df["station_code"] = station_codes

    scaler = StandardScaler()
    scaler.fit(train_df[param_names])

    joblib.dump(scaler, "data/std_scaler.joblib")

    train_df = apply_common_feature_engineering(
        df=train_df,
        param_names=param_names,
        lags=lags,
        scaler=scaler,
        station_to_int=station_to_int,
        map_station=False,
    )
    test_df = apply_common_feature_engineering(
        df=test_df,
        param_names=param_names,
        lags=lags,
        scaler=scaler,
        station_to_int=station_to_int,
    )
    val_df = apply_common_feature_engineering(
        df=val_df,
        param_names=param_names,
        lags=lags,
        scaler=scaler,
        station_to_int=station_to_int,
    )

    train_df = add_target_features(
        df=train_df,
        target_col=target_col,
        forecast_horizon=forecast_horizon,
    )
    test_df = add_target_features(
        df=test_df,
        target_col=target_col,
        forecast_horizon=forecast_horizon,
    )
    val_df = add_target_features(
        df=val_df, target_col=target_col, forecast_horizon=forecast_horizon
    )

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)


def prepare_base_measurements_df(
    param_ids: list[str],
    stations: list[str],
    start: datetime,
    end: datetime,
    source_dir: str,
) -> pd.DataFrame:

    start_ts = to_local_timestamp(start)
    end_ts = to_local_timestamp(end)
    monthly_periods = pd.period_range(start=start, end=end, freq="M")
    file_list: list[pd.DataFrame] = []
    for station in stations:
        print(f"Processing station {station}...")
        station_dfs = []
        for param_id in param_ids:
            files = [
                f"{source_dir}/{station}_hourly_param_{param_id}_{period.year}_{period.month:02d}.csv"
                for period in monthly_periods
            ]
            existing_files = [path for path in files if pd.io.common.file_exists(path)]
            if len(existing_files) == 0:
                continue
            param_dfs: list[pd.DataFrame] = []
            for f in existing_files:
                df0 = pd.read_csv(f, index_col="timestamp")
                param_dfs.append(df0)
            df_param = pd.concat(param_dfs, axis=0)
            df_param = df_param.groupby(df_param.index).mean(skipna=True)
            station_dfs.append(df_param)
        if len(station_dfs) == 0:
            continue
        df = pd.concat(station_dfs, axis=1, join = "outer")
        df = df.loc[df.index.to_series().between(start_ts, end_ts, inclusive="both")]
        if df.empty:
            continue
        df["station_id"] = station
        file_list.append(df)

    if len(file_list) == 0:
        return pd.DataFrame()

    measurements_df = pd.concat(file_list, ignore_index=False)
    measurements_df.rename(
        columns=dict(
            zip(param_ids, map_param_id_to_name(param_ids, name_type="nameInTable"))
        ),
        inplace=True,
    )
    measurements_df = measurements_df.reset_index().rename(columns={"index": "timestamp"})
    measurements_df = clean_and_resample(measurements_df)
    measurements_df = add_temporal_features(measurements_df)
    return measurements_df


def apply_common_feature_engineering(
    df: pd.DataFrame,
    param_names: list[str],
    lags: list[int],
    scaler: StandardScaler,
    station_to_int: dict[str, int],
    map_station: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    if map_station:
        df["station_code"] = df["station_id"].map(station_to_int)

    df = df[df["station_code"].notna()].copy()
    df["station_code"] = df["station_code"].astype(int)

    df[param_names] = scaler.transform(df[param_names])
    df = add_lag_features(df, param_names, lags)
    return df


def preprocess_inference_measurements(
    param_names: list[str],
    station_id: str,
    lags: list[int],
    start: datetime,
    end: datetime,
    station_mapping_path: str = "data/station_mapping.json",
    scaler_path: str = "data/std_scaler.joblib",
    source_dir: str = "data/temp",
) -> pd.DataFrame:
    param_ids = map_param_name_to_id(param_names)
    measurements_df = prepare_base_measurements_df(
        param_ids=param_ids,
        stations=[station_id],
        start=start,
        end=end,
        source_dir=source_dir,
    )
    if measurements_df.empty:
        return measurements_df

    with open(station_mapping_path, "r") as f:
        station_to_int = json.load(f)
    scaler = joblib.load(scaler_path)

    measurements_df = apply_common_feature_engineering(
        df=measurements_df,
        param_names=param_names,
        lags=lags,
        scaler=scaler,
        station_to_int=station_to_int,
    )
    return measurements_df


def select_stations(start: datetime, end: datetime) -> list[str]:
    stations_df = fetch_stations(force=False)
    start_timestamp = to_local_timestamp(start)
    end_timestamp = to_local_timestamp(end)
    stations_df = stations_df.loc[
        # (stations_df["Stationsumgebung"] == "städtisches Gebiet, Verkehr")
        (stations_df["messung_von"] <= start_timestamp)
        & (
            (stations_df["messung_bis"] >= end_timestamp)
            | (stations_df["messung_bis"].isna())
        )
    ]
    return stations_df["stationId"].astype(str).tolist()


def add_lag_features(
    df: pd.DataFrame, param_names: list[str], lags: list[int]
) -> pd.DataFrame:
    df = df.sort_values(["station_id", "datetime"])

    for lag in lags:
        for param in param_names:
            df[f"{param}_lag{lag}"] = df.groupby("station_id")[param].shift(lag)

    df.dropna(inplace=True)
    return df


def add_target_features(
    df: pd.DataFrame, target_col: str, forecast_horizon: int
) -> pd.DataFrame:
    df = df.sort_values(["station_id", "datetime"])
    for i in range(forecast_horizon):
        df[f"target_{target_col}_lag{i + 1}"] = df.groupby("station_id")[
            target_col
        ].shift(-i)
    df.dropna(inplace=True)
    return df


def retrieve_measurements(
    param_ids: list[str], stations: list[str], start: datetime, end: datetime
) -> None:
    for station in stations:
        available_param_ids, available_value_types = fetch_station_details(station)
        if not set(param_ids).issubset(set(available_param_ids)):
            print(
                f"Station {station} does not have all required parameters, skipping..."
            )
            continue
        if 2 not in available_value_types:
            print(available_value_types)
            print(f"Station {station} does not have hourly data, skipping...")
            continue

        print(f"Fetching data for station {station}...")
        fetch_hourly_measurements(
            station_id=station,
            start_year=start.year,
            start_month=start.month,
            end_year=end.year,
            end_month=end.month,
            param_ids=param_ids,
            force=False,
        )


def encode_cyclic_features(x: pd.Series, name: str, measurements_df: pd.DataFrame, max_value: float) -> pd.DataFrame:

    measurements_df[f"{name}_sin"] = np.sin(2 * np.pi * x / max_value)

    measurements_df[f"{name}_cos"] = np.cos(2 * np.pi * x / max_value)

    return measurements_df


def clean_and_resample(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(
        LOCAL_TZ
    )
    df = (
        df.set_index("datetime")
        .groupby("station_id")
        .resample("1h")
        .mean()
        .groupby(level="station_id")
        .ffill()
        .reset_index()
    )
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["datetime"].dt.day_of_week / 6

    df = encode_cyclic_features(df["datetime"].dt.hour, "hour", df, 24)
    df = encode_cyclic_features(df["datetime"].dt.dayofyear, "doy", df, 365)

    return df
