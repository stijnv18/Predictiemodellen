import csv
import pandas as pd

def check_missing_timestamps(file_name):
    df = pd.read_csv(file_name)
    df['timestamp'] = pd.to_datetime(df['utc_timestamp'])  # replace 'timestamp' with your column name
    df = df.sort_values('timestamp')
    missing_timestamps = pd.date_range(start = df['timestamp'].min(), end = df['timestamp'].max() ).difference(df['timestamp'])
    return missing_timestamps

missing_timestamps = check_missing_timestamps('residential4_grid_import_export_weather_fixed_timestamps.csv')  # replace 'your_file.csv' with your file name
if missing_timestamps.empty:
    print("No missing timestamps")
else:
    print("Missing timestamps:")
    print(missing_timestamps)