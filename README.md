# Air Quality

A project to monitor and analyze air quality metrics using open data from the [Luftmessnetz Hessen](https://www.hlnug.de/messwerte/datenportal).

## Features

- Real-time air data collection
- Machine learning model training for NO2 prediction
- TensorBoard integration for model monitoring

## API

Run the prediction API:

```bash
uvicorn api:app --reload
```

Endpoint:

- `GET /predict?station_id=0104&dt=2026-03-06T18:00:00&parameter=NO2`

The API normalizes `dt` to the top of the hour and returns the next forecast horizon.

## Dashboard

Run the Dash dashboard:

```bash
python dashboard.py
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050), set station/date/hour, and click **Predict** to fetch and visualize the forecast from the API.