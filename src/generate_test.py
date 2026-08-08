import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "lead1.0-small",
    "lead1.0-small.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "test_df.csv"
)
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


buildings = df["building_id"].unique()

print("Number of buildings:", len(buildings))
print("Building IDs:", buildings)

df["meter_reading"] = (
    df["meter_reading"]
    .ffill()
    .bfill()
)

sample_buildings = [
    941,
    623,
    1120,
    136,
    884,
    710,
    1143,
    879,
    117,
    948
]

test_df = df[
    df["building_id"].isin(sample_buildings)
].copy()


os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

test_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nTest dataset created successfully.")
print("Rows:", len(test_df))
print("Buildings included:", test_df["building_id"].unique())
print("Saved to:", OUTPUT_PATH)