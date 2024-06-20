import pandas as pd

# Load the energy consumption data
df_energy = pd.read_csv('processed_data.csv')

# Load the weather data
df_weather = pd.read_csv('hourly.csv')

# Convert 'DateTime' to datetime, make a copy, and set as index
df_energy['DateTime'] = pd.to_datetime(df_energy['DateTime'])
df_energy.set_index('DateTime', inplace=True)
df_energy.index = df_energy.index.tz_localize(None)

# Convert 'date' to datetime, make a copy, and set as index
df_weather['date'] = pd.to_datetime(df_weather['date'])
df_weather.set_index('date', inplace=True)
df_weather.index = df_weather.index.tz_localize(None)

# Resample the weather data to 30-minute frequency and interpolate missing values
df_weather = df_weather.resample('30T').interpolate()

# Merge the dataframes on the index
df_merged = pd.merge(df_energy, df_weather, left_index=True, right_index=True, how='inner')

df_merged = df_merged.drop('Unnamed: 0', axis=1)

# Reset the index to make 'DateTime' a column again
df_merged.reset_index(inplace=True)

# Rename the index column to 'DateTime'
df_merged.rename(columns={'index': 'DateTime'}, inplace=True)

# Save the merged dataframe to a new CSV file
df_merged.to_csv('merged_data.csv', index=False)