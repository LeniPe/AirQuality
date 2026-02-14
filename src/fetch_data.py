import json
from time import sleep
import requests
from datetime import datetime
import pandas as pd
import os


def fetch_hourly_measurements(
    station_id: str,
    start: datetime,
    end: datetime,
    param_ids: list[str],
    force: bool = False,
) -> None:

    url = "https://app.hlnug.de"

    start = int(start.timestamp())
    end = int(end.timestamp())
    param_ids_str = ",".join(param_ids)

    limit = 60 * 60 * 1000

    while end > start + limit:
        filename = f"data/raw/{station_id}_hourly_{start}_{start + limit}.csv"
        if not force and os.path.exists(filename):
            # print(f"File {filename} already exists, skipping...")
            start += limit
            continue

        # print(f"Fetching data from {start} to {start + limit}...")

        r = requests.get(
            f"{url}/json/lmw/getStationTableData/{station_id}/{param_ids_str}/{start}/{start + limit}?valueType=2"
        )
        r.raise_for_status()

        data = r.json()["data"]
        df = pd.DataFrame.from_dict(data, orient="index").dropna()
        if len(df) > 0:
            df.to_csv(filename, index=True, index_label="timestamp")
        start += limit
        sleep(1)
    return


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
