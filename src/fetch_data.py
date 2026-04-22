import json
from time import sleep
import requests
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
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


def _fetch_hourly_measurements_monthly(
    station_id: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    param_ids: list[str],
    force: bool = False,
    persist: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly measurements for a station across monthly periods.
    
    Args:
        station_id: Station ID
        start_year: Start year
        start_month: Start month (1-12)
        end_year: End year
        end_month: End month (1-12)
        param_ids: List of parameter IDs
        force: Force re-download even if files exist
        persist: Save to data/raw or data/temp
    """
    print(
        f"Fetching hourly measurements for station {station_id} from {start_year}-{start_month:02d} to {end_year}-{end_month:02d}..."
    )

    all_data = []
    
    # Iterate through each month in the range
    current = date(start_year, start_month, 1)
    end = date(end_year, end_month, 1)
    
    while current <= end:
        year = current.year
        month = current.month
        
        # Get the first and last day of the month
        month_start = date(year, month, 1)
        month_end = month_start + relativedelta(months=1) - relativedelta(days=1)
        
        # Convert to datetime at start and end of day
        start_datetime = datetime.combine(month_start, datetime.min.time())
        end_datetime = datetime.combine(month_end, datetime.max.time())
        
        start_timestamp = to_local_timestamp(start_datetime)
        end_timestamp = to_local_timestamp(end_datetime)
        
        # Fetch data for each parameter in this month
        for param_id in param_ids:
            month_str = f"{year}_{month:02d}"
            filename = (
                f"data/raw/{station_id}_hourly_param_{param_id}_{month_str}.csv"
                if persist
                else f"data/temp/{station_id}_hourly_param_{param_id}_{month_str}.csv"
            )
            
            # Check if file already exists
            if not force and os.path.exists(filename):
                print(f"File {filename} already exists, loading...")
                cached_df = pd.read_csv(filename)
                if len(cached_df) > 0:
                    all_data.append(cached_df)
                continue

            df = _request_hourly_measurements(
                station_id=station_id,
                param_id=param_id,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )
            
            if len(df) > 0:
                result_df = df.reset_index(names="timestamp")
                all_data.append(result_df)
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                result_df.to_csv(filename, index=False)
                print(f"Saved {len(result_df)} records to {filename}")
            else:
                print(f"No data available for {month_str}")
            
            sleep(1)
        
        # Move to next month
        current += relativedelta(months=1)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def fetch_hourly_measurements(
    station_id: str,
    start: datetime = None,
    end: datetime = None,
    param_ids: list[str] = None,
    start_year: int = None,
    start_month: int = None,
    end_year: int = None,
    end_month: int = None,
    force: bool = False,
    persist: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly measurements for a station.
    
    Supports two interfaces:
    1. Legacy datetime interface: provide start and end datetime objects
    2. Monthly interface: provide start_year, start_month, end_year, end_month
    
    Args:
        station_id: Station ID
        start: Start datetime (legacy interface)
        end: End datetime (legacy interface)
        param_ids: List of parameter IDs
        start_year: Start year (monthly interface)
        start_month: Start month (1-12) (monthly interface)
        end_year: End year (monthly interface)
        end_month: End month (1-12) (monthly interface)
        force: Force re-download even if files exist
        persist: Save to data/raw or data/temp
    """
    # Determine which interface is being used
    if start is not None and end is not None:
        # Legacy datetime interface - convert to monthly interface
        start_year = start.year
        start_month = start.month
        end_year = end.year
        end_month = end.month
    elif start_year is None or start_month is None or end_year is None or end_month is None:
        raise ValueError(
            "Either provide (start, end) datetime objects or "
            "(start_year, start_month, end_year, end_month) for monthly interface"
        )
    
    return _fetch_hourly_measurements_monthly(
        station_id=station_id,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        param_ids=param_ids,
        force=force,
        persist=persist,
    )


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
