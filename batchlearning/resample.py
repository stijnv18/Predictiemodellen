import pandas as pd

# Load the data
df = pd.read_csv('D:\\bachelor\\residential4_grid_import_export_weather_fixed_timestamps.csv')

# Convert 'utc_timestamp' to datetime format if it's not already
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])

# Set 'utc_timestamp' as the index
df = df.set_index('utc_timestamp')

# Resample to hourly frequency and calculate the mean
df_hourly = df.resample('H').mean()

# Reset the index
df_hourly = df_hourly.reset_index()

# Write the resampled DataFrame to a CSV file
df_hourly.to_csv('dataset_resampled_hourly.csv', index=False)

print(df_hourly.head(50))