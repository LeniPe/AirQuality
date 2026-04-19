import json
from src.validate import validate
from src.preprocessing import preprocess_measurements, select_stations
from src.train import train_model, feature_selection
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch

# Hyperparameters and configuration
num_epochs = 10
model_type = "quantile"  # "simple" or "quantile"
target_col = "no2"
forecast_horizon = 12
lr = 1e-3

def main():

    temporal_feature_cols = [
        "day_of_week",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]
    measurements_feature_cols = ["temp", "feuchte", "windge", "no2"]
    spatial_feature_cols = ["station_code"]
    lags = [1, 2, 3, 6, 12, 24]
    lag_feature_cols = []
    for lag in lags:
        lag_feature_cols += [f"{x}_lag{lag}" for x in measurements_feature_cols]
    feature_cols = temporal_feature_cols + spatial_feature_cols + lag_feature_cols
    print(f"Using features: {feature_cols}")

    target_cols = [f"target_{target_col}_lag{i + 1}" for i in range(forecast_horizon)]
    start, end = datetime(2025, 1, 1), datetime(2025, 12, 31)
    stations = select_stations(start=start, end=end)
    print(f"Selected {len(stations)} stations.")

    preprocess_measurements(
        param_names=measurements_feature_cols,
        stations=stations,
        lags=lags,
        start=start,
        end=end,
        target_col=target_col,
        forecast_horizon=forecast_horizon,
        retrieve_new_measurements=False,
    )
    select_features = False
    if select_features:
        selected_features = feature_selection(
            feature_cols=feature_cols, target_cols=target_cols, num_epochs=num_epochs
        )
        print(f"Selected features: {selected_features}")
    else:
        selected_features = feature_cols

    writer = SummaryWriter()
    config = {
        "lags": lags,
        "feature_cols": selected_features,
        "target_col": target_col,
        "forecast_horizon": forecast_horizon,
        "num_epochs": num_epochs,
        "model_type": model_type,
    }

    writer.add_text("config", json.dumps(config, indent=4))

    model, val_loss = train_model(
        feature_cols=selected_features,
        target_cols=target_cols,
        writer=writer,
        num_epochs=num_epochs,
        model_type=model_type,
        lr=lr,
    )

    writer.add_hparams(
        {
            "hparam/forecast_horizon": forecast_horizon, 
            "hparam/num_epochs": num_epochs,
            "hparam/lr": lr,
        },
        {
            "metric/val_loss": val_loss,
        },
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, "output/model_checkpoint.pth")
    validate(feature_cols=selected_features, target_cols=target_cols, model=model)


if __name__ == "__main__":
    main()
