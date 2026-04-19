from datetime import datetime, timedelta
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.fetch_data import fetch_hourly_measurements
from src.predict import load_model, predict_series
from src.preprocessing import map_param_name_to_id
from src.time_utils import to_local_datetime


class ObservationPoint(BaseModel):
    datetime: datetime
    value: float


class PredictionResponse(BaseModel):
    station_id: str
    forecast_origin_datetime: datetime
    parameter: str
    horizon_hours: int
    predictions: list[ObservationPoint]
    lower_quantile_predictions: list[ObservationPoint] | None = None
    upper_quantile_predictions: list[ObservationPoint] | None = None


class ObservationsResponse(BaseModel):
    station_id: str
    parameter: str
    observations: list[ObservationPoint]


app = FastAPI(title="AirQuality Prediction API")


@lru_cache(maxsize=1)
def get_model_bundle() -> tuple:
    return load_model()


@app.get("/predictions", response_model=PredictionResponse)
def predict_endpoint(
    station_id: str = Query(..., description="Station ID, e.g. 0104"),
    dt: datetime | None = Query(
        None,
        description="Forecast origin datetime in ISO format. Normalized to the start of the hour.",
    ),
    parameter: str = Query("NO2", description="Target parameter, currently NO2 only"),
) -> PredictionResponse:
    requested_dt = to_local_datetime(dt or datetime.now()).replace(
        minute=0, second=0, microsecond=0
    )

    model, feature_cols, _, target_col, lags, model_type = get_model_bundle()
    pred_times, pred_values, _ = predict_series(
        feature_cols=feature_cols,
        target_col=target_col,
        lags=lags,
        model=model,
        station_id=station_id,
        requested_dt=requested_dt,
        parameter=parameter,
    )

    forecast_points = [
        ObservationPoint(datetime=forecast_time, value=float(pred_value[1]))
        for forecast_time, pred_value in zip(pred_times, pred_values)
    ]
    lower_quantile_points = None
    upper_quantile_points = None

    if model_type == "quantile":
        lower_quantile_points = [
            ObservationPoint(datetime=forecast_time, value=float(pred_value[0]))
            for forecast_time, pred_value in zip(pred_times, pred_values)
        ]

        upper_quantile_points = [
            ObservationPoint(datetime=forecast_time, value=float(pred_value[2]))
            for forecast_time, pred_value in zip(pred_times, pred_values)
        ]

    return PredictionResponse(
        station_id=station_id,
        forecast_origin_datetime=requested_dt,
        parameter=target_col.upper(),
        horizon_hours=len(forecast_points),
        predictions=forecast_points,
        lower_quantile_predictions=lower_quantile_points,
        upper_quantile_predictions=upper_quantile_points,
    )


@app.get("/observations", response_model=ObservationsResponse)
def observations_endpoint(
    station_id: str = Query(..., description="Station ID, e.g. 0104"),
    parameter: str = Query("NO2", description="Parameter to fetch, e.g. NO2"),
    dt: datetime | None = Query(
        None,
        description="End datetime for observations in ISO format. Normalized to the start of the hour. Defaults to current time.",
    ),
    timedelta_hours: int = Query(
        12,
        description="Number of hours before the end datetime to fetch observations for.",
        ge=1,
        le=48,
    ),
):
    # This endpoint can be used by the dashboard to fetch observed data points for plotting
    end = to_local_datetime(dt or datetime.now()).replace(
        minute=0, second=0, microsecond=0
    )
    start = end - timedelta(hours=timedelta_hours)

    param_id = map_param_name_to_id([parameter.lower()])[0]
    if param_id is None:
        raise HTTPException(
            status_code=400, detail=f"Unsupported parameter '{parameter}'"
        )

    df = fetch_hourly_measurements(
        station_id=station_id,
        start=start,
        end=end,
        param_ids=[param_id],
        force=False,
        persist=False,
    )
    if df.empty:
        return ObservationsResponse(
            station_id=station_id, parameter=parameter, observations=[]
        )

    observations = []
    for _, row in df.iterrows():
        obs_time = datetime.fromtimestamp(int(row["timestamp"]))
        obs_value = row[param_id]
        observations.append(ObservationPoint(datetime=obs_time, value=obs_value))
    return ObservationsResponse(
        station_id=station_id, parameter=parameter, observations=observations
    )
