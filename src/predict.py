import os
import json
import joblib
import numpy as np
import pandas as pd

from huggingface_hub import hf_hub_download

MODEL_ID = os.environ.get(
    "HF_MODEL_ID",
    "Vikaash17/energy-consumption-anomaly-detection"
)

MODEL_DIR = "models"

MODEL_FILE = "rf_model.pkl"
SCALER_FILE = "scaler.pkl"
CONFIG_FILE = "config.json"


def download_model_files():
    print("Loading model files from Hugging Face...")
    print(f"Repository: {MODEL_ID}")

    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_FILE
    )

    scaler_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=SCALER_FILE
    )

    config_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=CONFIG_FILE
    )

    print("Model files downloaded successfully.")

    return model_path, scaler_path, config_path

def load_model():

    model_path, scaler_path, config_path = download_model_files()

    rf_model = joblib.load(model_path)
    scalers = joblib.load(scaler_path)

    with open(config_path, "r") as f:
        config = json.load(f)

    threshold = config.get("Threshold", 0.20)

    print(f"Threshold: {threshold}")
    print(f"Number of building scalers: {len(scalers)}")

    return rf_model, scalers, threshold


def preprocess(df, scalers):

    df = df.copy()
    if "timestamp" in df.columns:
        df["timeStamp"] = pd.to_datetime(df["timestamp"])

    elif "timeStamp" in df.columns:
        df["timeStamp"] = pd.to_datetime(df["timeStamp"])

    else:
        raise ValueError(
            "Input data must contain 'timestamp' or 'timeStamp'."
        )

    df = df.sort_values(
        ["building_id", "timeStamp"]
    ).copy()

    df["meter_reading"] = (
        df["meter_reading"]
        .ffill()
        .bfill()
    )

    df["hour"] = df["timeStamp"].dt.hour
    df["day"] = df["timeStamp"].dt.day_name()
    df["month"] = df["timeStamp"].dt.month

    df["meter_reading_scaled"] = np.nan

    for building_id, group in df.groupby("building_id"):

        if building_id not in scalers:
            raise ValueError(
                f"No scaler found for building_id={building_id}"
            )

        scaler = scalers[building_id]

        df.loc[
            df["building_id"] == building_id,
            "meter_reading_scaled"
        ] = scaler.transform(
            group["meter_reading"]
            .values
            .reshape(-1, 1)
        ).flatten()

    df["lag_1"] = (
        df.groupby("building_id")[
            "meter_reading_scaled"
        ].shift(1)
    )

    df["lag_2"] = (
        df.groupby("building_id")[
            "meter_reading_scaled"
        ].shift(2)
    )

    df["lag_3"] = (
        df.groupby("building_id")[
            "meter_reading_scaled"
        ].shift(3)
    )

    df["lag_24"] = (
        df.groupby("building_id")[
            "meter_reading_scaled"
        ].shift(24)
    )

    df["lag_48"] = (
        df.groupby("building_id")[
            "meter_reading_scaled"
        ].shift(48)
    )

    grouped = df.groupby("building_id")[
        "meter_reading_scaled"
    ]

    df["rolling_mean_24"] = grouped.transform(
        lambda x: x.rolling(24).mean()
    )

    df["rolling_std_24"] = grouped.transform(
        lambda x: x.rolling(24).std()
    )

    df["rolling_max_24"] = grouped.transform(
        lambda x: x.rolling(24).max()
    )

    df = df.dropna().copy()

    days_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    day_mapping = {
        day: index
        for index, day in enumerate(days_order)
    }

    df["day"] = df["day"].map(day_mapping)

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["day"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    feature_columns = [
        "building_id",
        "meter_reading_scaled",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_24",
        "lag_48",
        "rolling_mean_24",
        "rolling_std_24",
        "rolling_max_24",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos"
    ]

    X = df[feature_columns].copy()

    return df, X

def predict():

    rf_model, scalers, threshold = load_model()

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    test_path = os.path.join(
        project_root,
        "dataset",
        "test_df.csv"
    )

    print(f"\nLoading test data:")
    print(test_path)

    test_df = pd.read_csv(test_path)

    print(f"Input rows: {len(test_df)}")

    processed_df, X = preprocess(
        test_df,
        scalers
    )

    print(
        f"Rows after preprocessing: {len(X)}"
    )

    expected_features = list(
        rf_model.feature_names_in_
    )

    X = X[expected_features]

    probabilities = rf_model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    processed_df["anomaly_probability"] = probabilities
    processed_df["prediction"] = predictions

    print("\nPrediction completed.")

    print("\nPrediction counts:")
    print(
        processed_df["prediction"]
        .value_counts()
        .sort_index()
    )

    print("\nAnomaly percentage:")

    anomaly_percentage = (
        processed_df["prediction"].mean() * 100
    )

    print(
        f"{anomaly_percentage:.2f}%"
    )

    print("\nSample predictions:")

    print(
        processed_df[
            [
                "building_id",
                "meter_reading",
                "anomaly_probability",
                "prediction"
            ]
        ].head(20)
    )

    output_path = os.path.join(
        project_root,
        "dataset",
        "prediction_results.csv"
    )

    processed_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nPrediction results saved to:"
    )

    print(output_path)

if __name__ == "__main__":
    predict()