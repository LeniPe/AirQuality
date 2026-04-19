from datetime import datetime

import dash
from dash import Dash, Input, Output, State, dcc, html
import plotly.graph_objects as go
import requests

app: Dash = dash.Dash(__name__)
app.title = "Air Quality Forecast Dashboard"

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "24px"},
    children=[
        html.H1("Air Quality Forecast Dashboard"),
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                "gap": "12px",
            },
            children=[
                html.Div(
                    children=[
                        html.Label("Station ID"),
                        dcc.Input(
                            id="station-id",
                            type="text",
                            value="0104",
                            style={"width": "100%"},
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("Parameter"),
                        dcc.Dropdown(
                            id="parameter",
                            options=[{"label": "NO2", "value": "NO2"}],
                            value="NO2",
                            clearable=False,
                        ),
                    ]
                ),
                html.Div(
                    style={"display": "flex", "alignItems": "end"},
                    children=[
                        html.Button("Predict", id="predict-btn", n_clicks=0),
                    ],
                ),
            ],
        ),
        html.Hr(),
        html.Div(id="status", style={"whiteSpace": "pre-wrap", "marginBottom": "12px"}),
        dcc.Graph(id="forecast-graph"),
    ],
)


def _fetch_data(station_id: str, parameter: str, type: str):
    if type not in {"observations", "predictions"}:
        raise ValueError(
            f"Invalid type '{type}' for data fetch. Expected 'observations' or 'predictions'."
        )
    api_base_url = "http://localhost:8000"
    endpoint = f"{api_base_url.rstrip('/')}/{type}"
    try:
        response = requests.get(
            endpoint,
            params={
                "station_id": station_id.strip(),
                "parameter": parameter,
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch {type}: {exc}") from exc

    payload = response.json()
    return payload


@app.callback(
    Output("status", "children"),
    Output("forecast-graph", "figure"),
    Input("predict-btn", "n_clicks"),
    State("station-id", "value"),
    State("parameter", "value"),
)
def run_prediction(
    n_clicks: int,
    station_id: str,
    parameter: str,
    api_base_url: str = "http://localhost:8000",
):
    fig = go.Figure()
    fig.update_layout(
        title="Air Quality Forecast",
        xaxis_title="Datetime",
        yaxis_title="Predicted Value",
        template="plotly_white",
    )

    if n_clicks == 0:
        return "Click 'Predict' to request a forecast.", fig

    if not station_id or not station_id.strip():
        return "Station ID is required.", fig

    payload = _fetch_data(
        station_id=station_id, parameter=parameter, type="predictions"
    )
    predictions = payload.get("predictions", [])

    if not predictions:
        return "No predictions returned by API.", fig

    x_values = [p["datetime"] for p in predictions]
    y_values = [float(p["value"]) for p in predictions]

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            name="Forecast",
        )
    )

    if "lower_quantile_predictions" in payload and payload["lower_quantile_predictions"]:
        x_lower = [p["datetime"] for p in payload["lower_quantile_predictions"]]
        y_lower = [float(p["value"]) for p in payload["lower_quantile_predictions"]]
        fig.add_trace(
            go.Scatter(
                x=x_lower,
                y=y_lower,
                mode="lines",
                name="Lower Quantile",
                line=dict(dash="dash", color="red"),
            )
        )
    if "upper_quantile_predictions" in payload and payload["upper_quantile_predictions"]:
        x_upper = [p["datetime"] for p in payload["upper_quantile_predictions"]]
        y_upper = [float(p["value"]) for p in payload["upper_quantile_predictions"]]
        fig.add_trace(
            go.Scatter(
                x=x_upper,
                y=y_upper,
                mode="lines",
                name="Upper Quantile",
                line=dict(dash="dash", color="red"),
            )
        )

    status = (
        f"Station: {payload['station_id']}\n"
        f"Parameter: {payload['parameter']}\n"
        f"Forecast Origin: {datetime.fromisoformat(payload['forecast_origin_datetime']).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Forecast Horizon: {payload['horizon_hours']} hours"
    )

    payload_obs = _fetch_data(
        station_id=station_id, parameter=parameter, type="observations"
    )
    observations = payload_obs.get("observations", [])
    if observations:
        x_obs = [o["datetime"] for o in observations]
        y_obs = [float(o["value"]) for o in observations]
        fig.add_trace(
            go.Scatter(
                x=x_obs,
                y=y_obs,
                mode="lines+markers",
                name="Observations",
            )
        )
    return status, fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
