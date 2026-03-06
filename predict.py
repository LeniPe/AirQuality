from src.predict import predict, load_model

model, feature_cols, target_cols, target_col, lags = load_model()
print(feature_cols)
predict(feature_cols, target_col, lags, model)
