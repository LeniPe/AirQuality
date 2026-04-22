import json
import sqlite3
from time import sleep
import requests
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
from src.time_utils import to_local_timestamp


MEASUREMENT_SCHEMA: dict[str, str] = {
    "timestamp": "int64",
    "station_id": "string",
    "param_id": "string",
    "value": "float64",
    "year": "int64",
    "month": "int64",
}


def _enforce_measurement_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    normalized = df.copy()

    if "timestamp" in normalized.columns:
        normalized["timestamp"] = pd.to_numeric(normalized["timestamp"], errors="coerce")
    if "value" in normalized.columns:
        normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    if "year" in normalized.columns:
        normalized["year"] = pd.to_numeric(normalized["year"], errors="coerce")
    if "month" in normalized.columns:
        normalized["month"] = pd.to_numeric(normalized["month"], errors="coerce")

    required_numeric = [
        col for col in ["timestamp", "value", "year", "month"] if col in normalized.columns
    ]
    if len(required_numeric) > 0:
        normalized = normalized.dropna(subset=required_numeric)

    for col, dtype in MEASUREMENT_SCHEMA.items():
        if col not in normalized.columns:
            continue
        if dtype == "string":
            normalized[col] = normalized[col].astype(str)
        else:
            normalized[col] = normalized[col].astype(dtype)

    return normalized


def _manifest_path(base_dir: str) -> str:
    return os.path.join(base_dir, "chunk_manifest.sqlite")


def _parquet_root(base_dir: str) -> str:
    return os.path.join(base_dir, "parquet")


def _ensure_manifest_table(manifest_path: str) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with sqlite3.connect(manifest_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_manifest (
                station_id TEXT NOT NULL,
                param_id TEXT NOT NULL,
                year INTEGER NOT NULL,
                month INTEGER NOT NULL,
                status TEXT NOT NULL,
                row_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (station_id, param_id, year, month)
            )
            """
        )


def _is_chunk_complete(
    manifest_path: str,
    station_id: str,
    param_id: str,
    year: int,
    month: int,
) -> bool:
    with sqlite3.connect(manifest_path) as connection:
        row = connection.execute(
            """
            SELECT status
            FROM chunk_manifest
            WHERE station_id = ? AND param_id = ? AND year = ? AND month = ?
            """,
            (station_id, param_id, year, month),
        ).fetchone()
    return row is not None and row[0] == "complete"


def _update_manifest_chunk(
    manifest_path: str,
    station_id: str,
    param_id: str,
    year: int,
    month: int,
    row_count: int,
    status: str = "complete",
) -> None:
    updated_at = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(manifest_path) as connection:
        connection.execute(
            """
            INSERT INTO chunk_manifest (
                station_id, param_id, year, month, status, row_count, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(station_id, param_id, year, month) DO UPDATE SET
                status = excluded.status,
                row_count = excluded.row_count,
                updated_at = excluded.updated_at
            """,
            (station_id, param_id, year, month, status, row_count, updated_at),
        )


def _extract_long_measurements(df: pd.DataFrame, param_id: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "param_id", "value"])

    result_df = df.reset_index(names="timestamp")
    value_column = param_id if param_id in result_df.columns else None
    if value_column is None:
        candidate_columns = [col for col in result_df.columns if col != "timestamp"]
        if len(candidate_columns) == 0:
            return pd.DataFrame(columns=["timestamp", "param_id", "value"])
        value_column = candidate_columns[0]

    values = result_df[["timestamp", value_column]].rename(columns={value_column: "value"})
    values["param_id"] = str(param_id)
    values = values[["timestamp", "param_id", "value"]]
    return _enforce_measurement_schema(values)


def _read_cached_parquet_chunk(
    parquet_root: str,
    station_id: str,
    param_id: str,
    year: int,
    month: int,
) -> pd.DataFrame:
    if not os.path.exists(parquet_root):
        return pd.DataFrame(columns=["timestamp", "station_id", "param_id", "value"])

    filters = [
        ("station_id", "==", station_id),
        ("param_id", "==", str(param_id)),
        ("year", "==", year),
        ("month", "==", month),
    ]
    try:
        cached = pd.read_parquet(parquet_root, filters=filters)
        return _enforce_measurement_schema(cached)
    except (FileNotFoundError, ValueError, OSError):
        return pd.DataFrame(columns=["timestamp", "station_id", "param_id", "value"])


def _append_to_parquet_dataset(parquet_root: str, long_df: pd.DataFrame) -> None:
    if long_df.empty:
        return
    os.makedirs(parquet_root, exist_ok=True)
    long_df = _enforce_measurement_schema(long_df)
    long_df.to_parquet(
        parquet_root,
        index=False,
        partition_cols=["year", "month"],
        engine="pyarrow",
    )


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

    base_dir = "data/raw" if persist else "data/temp"
    parquet_root = _parquet_root(base_dir)
    manifest_path = _manifest_path(base_dir)
    _ensure_manifest_table(manifest_path)

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
            if not force and _is_chunk_complete(
                manifest_path, station_id, str(param_id), year, month
            ):
                cached_df = _read_cached_parquet_chunk(
                    parquet_root=parquet_root,
                    station_id=station_id,
                    param_id=str(param_id),
                    year=year,
                    month=month,
                )
                if len(cached_df) > 0:
                    all_data.append(cached_df)
                continue

            month_str = f"{year}_{month:02d}"
            df = _request_hourly_measurements(
                station_id=station_id,
                param_id=param_id,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )
            long_df = _extract_long_measurements(df, str(param_id))

            if len(long_df) > 0:
                long_df["station_id"] = station_id
                long_df["year"] = year
                long_df["month"] = month
                long_df = long_df[
                    ["timestamp", "station_id", "param_id", "value", "year", "month"]
                ]
                long_df = _enforce_measurement_schema(long_df)
                _append_to_parquet_dataset(parquet_root, long_df)
                _update_manifest_chunk(
                    manifest_path=manifest_path,
                    station_id=station_id,
                    param_id=str(param_id),
                    year=year,
                    month=month,
                    row_count=len(long_df),
                )
                all_data.append(long_df.drop(columns=["year", "month"]))
                print(
                    f"Saved {len(long_df)} records for station={station_id}, "
                    f"param={param_id}, month={month_str}"
                )
            else:
                _update_manifest_chunk(
                    manifest_path=manifest_path,
                    station_id=station_id,
                    param_id=str(param_id),
                    year=year,
                    month=month,
                    row_count=0,
                )
                print(f"No data available for {month_str}, param_id={param_id}")

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
