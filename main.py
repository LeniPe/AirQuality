from src.preprocessing import preprocess_measurements, select_stations
from src.train import train_model
from src.predict import predict
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def main():

    temporal_feature_cols = [
        "day_of_week",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]
    measurements_feature_cols = ["temp", "feuchte", "windge", "no2"]
    lags = [1]
    lag_feature_cols = []
    for lag in lags:
        lag_feature_cols += [f"{x}_lag{lag}" for x in measurements_feature_cols]
    feature_cols = temporal_feature_cols + lag_feature_cols
    print(f"Using features: {feature_cols}")
    target_col = "no2"

    start, end = datetime(2025, 1, 1), datetime(2025, 12, 31)
    stations = select_stations(start=start, end=end)
    print(f"Selected {len(stations)} stations.")

    preprocess_measurements(
        param_names=measurements_feature_cols,
        stations=stations,
        lags=lags,
        start=start,
        end=end,
        retrieve_new_measurements=False,
    )

    writer = SummaryWriter()

    model = train_model(feature_cols, target_col, writer=writer, num_epochs=10)

    predict(feature_cols, target_col, model, writer=writer)


if __name__ == "__main__":
    main()
