import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from pandas.api.types import CategoricalDtype

from data_preprocess import preprocess

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "normalized_dataset"
)

RAW_DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "lead1.0-small",
    "lead1.0-small.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "rf_model.pkl"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "config.json"
)

TEST_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "test_df.csv"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)

df = df.sort_values(
    ["building_id", "timeStamp"]
).reset_index(drop=True)

print("\nCreating lag features...")

grouped_reading = df.groupby(
    "building_id"
)["meter_reading_scaled"]

df["lag_1"] = grouped_reading.shift(1)
df["lag_2"] = grouped_reading.shift(2)
df["lag_3"] = grouped_reading.shift(3)
df["lag_24"] = grouped_reading.shift(24)
df["lag_48"] = grouped_reading.shift(48)

print("Creating rolling features...")

df["rolling_mean_24"] = (
    grouped_reading
    .transform(lambda x: x.rolling(24).mean())
)

df["rolling_std_24"] = (
    grouped_reading
    .transform(lambda x: x.rolling(24).std())
)

df["rolling_max_24"] = (
    grouped_reading
    .transform(lambda x: x.rolling(24).max())
)

df = df.dropna().reset_index(drop=True)


days_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_type = CategoricalDtype(
    categories=days_order,
    ordered=True
)

df["day"] = df["day"].astype(day_type)
df["day"] = df["day"].cat.codes


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

FEATURE_COLUMNS = [
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

X = df[FEATURE_COLUMNS]

y = df["anomaly"]


print("\nFeature columns:")
print(X.columns.tolist())

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestClassifier(
    class_weight={
        0: 1,
        1: 10
    },
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

print("Training completed.")

print("\n" + "=" * 60)
print("THRESHOLD EVALUATION")
print("=" * 60)

y_proba = rf_model.predict_proba(
    X_test
)[:, 1]

for threshold in [
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
    0.20
]:

    y_pred = (
        y_proba >= threshold
    ).astype(int)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]

    print(
        f"Threshold {threshold:.2f} → "
        f"Precision: {precision:.2f}, "
        f"Recall: {recall:.2f}, "
        f"F1: {f1:.2f}"
    )

OPTIMAL_THRESHOLD = 0.20

y_pred = (
    y_proba >= OPTIMAL_THRESHOLD
).astype(int)


print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

config = {
    "threshold": OPTIMAL_THRESHOLD,
    "historicWindow": 40,
    "model_version": 1.0
}

with open(
    CONFIG_PATH,
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )

print("\nConfiguration saved to:")
print(CONFIG_PATH)

if os.path.exists(MODEL_PATH):

    print("\nRandom Forest model already exists.")
    print("Keeping existing model:")
    print(MODEL_PATH)

else:

    joblib.dump(
        rf_model,
        MODEL_PATH
    )

    print("\nRandom Forest model saved to:")
    print(MODEL_PATH)

if not os.path.exists(TEST_DATA_PATH):

    sample_test_df = pd.read_csv(
        RAW_DATASET_PATH
    )

    sample_test_df = (
        sample_test_df[
            sample_test_df["building_id"] == 32
        ]
        .head(10000)
    )

    sample_test_df.to_csv(
        TEST_DATA_PATH,
        index=False
    )

    print("\nSample test dataset saved to:")
    print(TEST_DATA_PATH)

else:

    print("\nSample test dataset already exists.")
    print("Keeping existing file.")

print("\n" + "=" * 60)
print("BUILDING-LEVEL EVALUATION")
print("=" * 60)

test_buildings = X_test["building_id"]

evaluation_features = [
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

results = []

for building_id in test_buildings.unique():

    mask = (
        test_buildings == building_id
    )

    building_X = X_test.loc[
        mask,
        evaluation_features
    ]

    building_y = y_test.loc[mask]

    if building_y.sum() == 0:
        continue

    building_proba = (
        rf_model
        .predict_proba(building_X)[:, 1]
    )

    building_pred = (
        building_proba >= OPTIMAL_THRESHOLD
    ).astype(int)

    building_f1 = f1_score(
        building_y,
        building_pred,
        zero_division=0
    )

    results.append({
        "building_id": building_id,
        "f1_score": round(
            building_f1,
            2
        ),
        "anomaly_count": int(
            building_y.sum()
        )
    })


results_df = (
    pd.DataFrame(results)
    .sort_values(
        "f1_score",
        ascending=False
    )
)


if not results_df.empty:

    print("\nTop 10 buildings:")
    print(
        results_df
        .head(10)
        .to_string(index=False)
    )

    print("\nBottom 10 buildings:")
    print(
        results_df
        .tail(10)
        .to_string(index=False)
    )

    print(
        f"\nMean F1: "
        f"{results_df['f1_score'].mean():.2f}"
    )

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("Model:", MODEL_PATH)
print("Scaler:", SCALER_PATH)
print("Config:", CONFIG_PATH)
print("Threshold:", OPTIMAL_THRESHOLD)
print("Number of features:", len(FEATURE_COLUMNS))