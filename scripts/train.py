import json

import torch
from torch.utils.tensorboard import SummaryWriter

from config import TrainConfig
from airquality.training.evaluate import evaluate
from airquality.data.preprocessing import preprocess_measurements, select_stations
from airquality.training.train import train_model, feature_selection


def main():
    cfg = TrainConfig()

    print(f"Using features: {cfg.feature_cols}")

    stations = select_stations(start=cfg.start, end=cfg.end)
    print(f"Selected {len(stations)} stations.")

    preprocess_measurements(
        param_names=cfg.measurements_feature_cols,
        stations=stations,
        lags=cfg.lags,
        start=cfg.start,
        end=cfg.end,
        target_col=cfg.target_col,
        forecast_horizon=cfg.forecast_horizon,
        retrieve_new_measurements=cfg.retrieve_new_measurements,
    )

    if cfg.select_features:
        selected_features = feature_selection(
            feature_cols=cfg.feature_cols,
            target_cols=cfg.target_cols,
            num_epochs=cfg.num_epochs,
        )
        print(f"Selected features: {selected_features}")
    else:
        selected_features = cfg.feature_cols

    writer = SummaryWriter()
    run_config = {
        "lags": cfg.lags,
        "feature_cols": selected_features,
        "target_col": cfg.target_col,
        "forecast_horizon": cfg.forecast_horizon,
        "num_epochs": cfg.num_epochs,
        "model_type": cfg.model_type,
    }

    writer.add_text("config", json.dumps(run_config, indent=4))

    model, val_loss = train_model(
        feature_cols=selected_features,
        target_cols=cfg.target_cols,
        writer=writer,
        num_epochs=cfg.num_epochs,
        model_type=cfg.model_type,
        lr=cfg.lr,
    )

    writer.add_hparams(
        {
            "hparam/forecast_horizon": cfg.forecast_horizon,
            "hparam/num_epochs": cfg.num_epochs,
            "hparam/lr": cfg.lr,
        },
        {
            "metric/val_loss": val_loss,
        },
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": run_config,
    }
    torch.save(checkpoint, "output/model_checkpoint.pth")
    evaluate(feature_cols=selected_features, target_cols=cfg.target_cols, model=model)


if __name__ == "__main__":
    main()
