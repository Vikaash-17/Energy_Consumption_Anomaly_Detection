import pandas as pd
import numpy as np

def calculate_statistics(df):
    total_count   = len(df)
    anomaly_count = int((df['prediction'] == 1).sum())
    normal_count  = int((df['prediction'] == 0).sum())
    anomaly_rate  = round(anomaly_count / total_count * 100, 2)
    return total_count, anomaly_count, normal_count, anomaly_rate

def consumption_statistics(df):
    mean   = round(df['meter_reading'].mean(), 2)
    median = round(df['meter_reading'].median(), 2)
    min_val = round(df['meter_reading'].min(), 2)
    max_val = round(df['meter_reading'].max(), 2)
    return mean, median, min_val, max_val

def chart_data(df):
    total_records = len(df)
    anomaly_count = int((df['prediction'] == 1).sum())
    normal_count  = int((df['prediction'] == 0).sum())
    labels      = ['Anomaly', 'Normal']
    values      = [anomaly_count, normal_count]
    percentages = [
        round(anomaly_count / total_records * 100, 2),
        round(normal_count  / total_records * 100, 2)
    ]
    return labels, values, percentages

def top_anomalies(df):
    return (
        df[df['prediction'] == 1]
        .sort_values('meter_reading', ascending=False)
        [['timestamp', 'building_id', 'meter_reading']]
        .head(10)
        .to_dict('records')
    )

def timeline(df):
    labels      = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    values      = df['meter_reading'].tolist()
    predictions = df['prediction'].tolist()
    return labels, values, predictions

def anomalies_by_building(df):
    return (
        df[df['prediction'] == 1]
        .groupby('building_id')
        .size()
        .reset_index(name='anomaly_count')
        .sort_values('anomaly_count', ascending=False)
        .to_dict('records'))