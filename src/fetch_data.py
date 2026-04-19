import json
from time import sleep
import requests
from datetime import datetime
import pandas as pd
import os
from src.time_utils import to_local_timestamp


def _request_hourly_measurements(
    station_id: str,
    param_id: str,
    start_timestamp: int,
    end_timestamp: int,
) -> pd.DataFrame:
    url = "https://app.hlnug.de"
    response = requests.get(
        f"{url}/json/lmw/getStationTableData/{station_id}/{param_id}/{start_timestamp}/{end_timestamp}?valueType=2"
    )
    response.raise_for_status()

    data = response.json()["data"]
    return pd.DataFrame.from_dict(data, orient="index").dropna()


def fetch_hourly_measurements(
    station_id: str,
    start: datetime,
    end: datetime,
    param_ids: list[str],
    force: bool = False,
    persist: bool = True,
) -> pd.DataFrame:

    start_timestamp = to_local_timestamp(start)
    end_timestamp = to_local_timestamp(end)

    limit = 60 * 60 * 1000
    print(
        f"Fetching hourly measurements for station {station_id} from {start} to {end}..."
    )

    all_data = []
    while start_timestamp < end_timestamp:
        chunk_end = min(start_timestamp + limit, end_timestamp)
        for param_id in param_ids:
            filename = (
                f"data/raw/{station_id}_hourly_param{param_id}_{start_timestamp}_{chunk_end}.csv"
                if persist
                else f"data/temp/{station_id}_hourly_param{param_id}_{start_timestamp}_{chunk_end}.csv"
            )

            if not force and os.path.exists(filename):
                print(f"File {filename} already exists, skipping...")
                cached_df = pd.read_csv(filename)
                if len(cached_df) > 0:
                    all_data.append(cached_df)
                continue

            df = _request_hourly_measurements(
                station_id=station_id,
                param_id=param_id,
                start_timestamp=start_timestamp,
                end_timestamp=chunk_end,
            )
            if len(df) > 0:
                result_df = df.reset_index(names="timestamp")
                all_data.append(result_df)
                df.to_csv(filename, index=True, index_label="timestamp")

            sleep(1)

        start_timestamp = chunk_end

    if all_data:
        return pd.concat(all_data)
    return pd.DataFrame()


def fetch_parameters(force=False) -> pd.DataFrame:
    filename = "data/parameters.csv"
    if not force and os.path.exists(filename):
        return pd.read_csv(filename)

    url = "https://app.hlnug.de"
    r = requests.get(f"{url}/json/lmw/getThemeParameters/1")
    r.raise_for_status()
    params = pd.DataFrame(r.json())
    params = params[["paramId", "displayName", "shortName", "nameInTable", "unit"]]
    params.to_csv(filename, index=False)
    return params


def fetch_stations(force=False) -> pd.DataFrame:
    filename = "data/stations.csv"
    if not force and os.path.exists(filename):
        return pd.read_csv(filename)

    url = "https://app.hlnug.de"
    r = requests.get(f"{url}/json/lmw/getThemeStations/1")
    r.raise_for_status()
    stations = pd.DataFrame(r.json())
    stations["Stationsumgebung"] = stations.baseInformation.apply(
        lambda x: json.loads(x)["singleselect-1"]["value"]
    )

    stations = stations[
        [
            "stationId",
            "displayName",
            "lat",
            "lon",
            "Stationsumgebung",
            "messung_von",
            "messung_bis",
        ]
    ]
    print(stations.head())
    stations.to_csv(filename, index=False)
    return stations


def fetch_station_details(station_id: str, force=False) -> list:
    url = "https://app.hlnug.de"
    r = requests.get(f"{url}/json/lmw/getStation/{station_id}?join=")
    r.raise_for_status()
    measured_parameters = r.json()["parameters"]
    parameter_ids = [str(param["paramId"]) for param in measured_parameters]

    # get station value types
    r = requests.get(f"{url}/json/lmw/getStationValueTypes/{station_id}/1")
    r.raise_for_status()
    value_types = r.json()

    return [parameter_ids, value_types]
