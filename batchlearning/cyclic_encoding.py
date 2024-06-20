import pandas as pd
import numpy as np
df = pd.read_csv('D:\\bachelor\\residential4_grid_import_export_weather_fixed_timestamps.csv')
# Convert 'utc_timestamp' to datetime format if it's not already
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])

# Extract hour from 'utc_timestamp'
df['hour'] = df['utc_timestamp'].dt.hour

# Perform cyclic encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)

# Drop the original 'hour' column
df = df.drop('hour', axis=1)
print(df.head(50))