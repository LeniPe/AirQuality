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
from glob import glob
import numpy as np


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
    df: pd.DataFrame, test_size: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame]:

    timestamps = df["datetime"].sort_values().unique()
    split_index = int(len(timestamps) * (1 - test_size))
    train_df = df[df["datetime"].isin(timestamps[:split_index])]
    test_df = df[df["datetime"].isin(timestamps[split_index:])]
    print(
        f"Train set: {train_df['datetime'].min()} - {train_df['datetime'].max()} with {len(train_df.station_id.unique())} stations"
    )
    print(
        f"Test set: {test_df['datetime'].min()} - {test_df['datetime'].max()} with {len(test_df.station_id.unique())} stations"
    )
    return train_df, test_df


def preprocess_measurements(
    param_names: list[str],
    stations: list[str],
    lags: list[int],
    start: datetime,
    end: datetime,
    retrieve_new_measurements: bool = False,
) -> None:

    param_ids = map_param_name_to_id(param_names)

    if retrieve_new_measurements:
        retrieve_measurements(param_ids, stations, start, end)

    file_list: list[pd.DataFrame] = []
    for station in stations:
        files = glob(f"data/raw/{station}_hourly_*.csv")
        if len(files) == 0:
            continue
        for f in files:
            df = pd.read_csv(
                f, usecols=["timestamp"] + param_ids, dtype={"timestamp": int}
            )
            df = df.loc[
                df.timestamp.between(int(start.timestamp()), int(end.timestamp()))
            ]
            df["station_id"] = station
            file_list.append(df)

    measurements_df = pd.concat(file_list, ignore_index=True)
    measurements_df.rename(
        columns=dict(
            zip(param_ids, map_param_id_to_name(param_ids, name_type="nameInTable"))
        ),
        inplace=True,
    )

    measurements_df.drop_duplicates(inplace=True)

    measurements_df["datetime"] = pd.to_datetime(measurements_df["timestamp"], unit="s")

    measurements_df = (
        measurements_df.set_index("datetime")
        .groupby("station_id")
        .resample("1h")
        .mean()
        .reset_index()
    )

    measurements_df["day_of_week"] = measurements_df["datetime"].dt.day_of_week / 6

    measurements_df = encode_cyclic_features(
        measurements_df["datetime"].dt.hour, "hour", measurements_df
    )
    measurements_df = encode_cyclic_features(
        measurements_df["datetime"].dt.dayofyear, "doy", measurements_df
    )

    train_df, test_df = train_test_split(measurements_df, test_size=0.2)

    station_codes, station_index = pd.factorize(train_df["station_id"])
    station_to_int = {station: i for i, station in enumerate(station_index)}
    with open("data/station_mapping.json", "w") as f:
        json.dump(station_to_int, f)

    train_df["station_code"] = station_codes
    test_df["station_code"] = test_df["station_id"].map(station_to_int)

    scaler = StandardScaler()
    scaler.fit(train_df[param_names])

    train_df[param_names] = scaler.transform(train_df[param_names])
    test_df[param_names] = scaler.transform(test_df[param_names])

    train_df = add_lag_features(param_names, train_df, lags)
    test_df = add_lag_features(param_names, test_df, lags)

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)


def select_stations(start: datetime, end: datetime) -> list[str]:
    stations_df = fetch_stations(force=False)
    start = int(start.timestamp())
    end = int(end.timestamp())
    stations_df = stations_df.loc[
        # (stations_df["Stationsumgebung"] == "städtisches Gebiet, Verkehr")
        (stations_df["messung_von"] <= start)
        & ((stations_df["messung_bis"] >= end) | (stations_df["messung_bis"].isna()))
    ]
    return stations_df["stationId"].astype(str).tolist()


def add_lag_features(param_names, measurements_df, lags):
    measurements_df = measurements_df.sort_values(["station_id", "datetime"])

    for lag in lags:
        for param in param_names:
            measurements_df[f"{param}_lag{lag}"] = measurements_df.groupby(
                "station_id"
            )[param].shift(lag)

    measurements_df.dropna(inplace=True)
    return measurements_df


def retrieve_measurements(param_ids, stations, start, end):
    for station in stations:
        available_param_ids, available_value_types = fetch_station_details(station)
        if not set(param_ids).issubset(set(available_param_ids)):
            print(
                f"Station {station} does not have all required parameters, skipping..."
            )
            continue
        if not 2 in available_value_types:
            print(available_value_types)
            print(f"Station {station} does not have hourly data, skipping...")
            continue

        print(f"Fetching data for station {station}...")
        fetch_hourly_measurements(
            station,
            start,
            end,
            param_ids,
            force=False,
        )


def encode_cyclic_features(x: pd.Series, name: str, measurements_df: pd.DataFrame):
    """
    Encode a cyclic feature into sine and cosine components.

    This function transforms a cyclic variable (e.g., hour of day, day of week, month)
    into its sine and cosine representations, which helps machine learning models
    understand the circular nature of the data.

    Parameters
    ----------
    x : pd.Series
        The cyclic feature values to be encoded.
    name : str
        The base name for the output columns. Two columns will be created:
        '{name}_sin' and '{name}_cos'.
    measurements_df : pd.DataFrame
        The DataFrame to which the encoded features will be added as new columns.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with two new columns containing the sine and cosine
        encoded features, rounded to 2 decimal places.
    """

    max_value = x.max()
    measurements_df[f"{name}_sin"] = np.round(
        np.sin(2 * np.pi * x / max_value), decimals=2
    )

    measurements_df[f"{name}_cos"] = np.round(
        np.cos(2 * np.pi * x / max_value), decimals=2
    )

    return measurements_df
