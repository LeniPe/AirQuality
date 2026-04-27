from datetime import datetime


LOCAL_TZ = datetime.now().astimezone().tzinfo


def to_local_datetime(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=LOCAL_TZ)
    return dt.astimezone(LOCAL_TZ)


def to_local_timestamp(dt: datetime) -> int:
    return int(to_local_datetime(dt).timestamp())


def from_local_timestamp(ts: int | float) -> datetime:
    return datetime.fromtimestamp(ts, tz=LOCAL_TZ)
