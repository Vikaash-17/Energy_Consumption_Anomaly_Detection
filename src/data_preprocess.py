import pandas as pd
import numpy as np
import joblib
with open(r'D:\\Machine Learning\\projects\\enegy consumption anamoly detection\\models\scaler.pkl', 'rb') as f:
    scalers = joblib.load(f)


def preprocess(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['building_id', 'timestamp'])
    for b_id, group in df.groupby('building_id'):
        if b_id not in scalers:
            continue
        scaler = scalers[b_id]
        df.loc[group.index, 'meter_reading_scaled'] = scaler.transform(
            group['meter_reading'].values.reshape(-1, 1)
        ).flatten()

    df['hour']  = df['timestamp'].dt.hour
    df['day']   = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month

    df['lag_1']  = df.groupby('building_id')['meter_reading'].shift(1)
    df['lag_2']  = df.groupby('building_id')['meter_reading'].shift(2)
    df['lag_3']  = df.groupby('building_id')['meter_reading'].shift(3)
    df['lag_24'] = df.groupby('building_id')['meter_reading'].shift(24)
    df['lag_48'] = df.groupby('building_id')['meter_reading'].shift(48)

    df['rolling_mean_24'] = df.groupby('building_id')['meter_reading'].rolling(24).mean().reset_index(0, drop=True)
    df['rolling_std_24']  = df.groupby('building_id')['meter_reading'].rolling(24).std().reset_index(0, drop=True)
    df['rolling_max_24']  = df.groupby('building_id')['meter_reading'].rolling(24).max().reset_index(0, drop=True)

    df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin']   = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos']   = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df = df.dropna()

    features = [
        'meter_reading_scaled',
        'lag_1', 'lag_2', 'lag_3',
        'lag_24', 'lag_48',
        'rolling_mean_24', 'rolling_std_24', 'rolling_max_24',
        'hour_sin', 'hour_cos',
        'day_sin', 'day_cos',
        'month_sin', 'month_cos'
    ]

    return df[features]