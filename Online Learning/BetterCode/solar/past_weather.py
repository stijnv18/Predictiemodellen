# Import Meteostat library and dependencies
from datetime import datetime, timedelta
from meteostat import Point, Daily
import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame and 'consumption' is your target variable
df = pd.read_csv('combined.csv')

# Set time period to the past week
end = df["datetime"].max()
start = df["datetime"].min()
# Create point for sydney -33.84721 151.11638
location = Point(33.84721, 151.11638, 0)

# Get daily data for the past week
data = Daily(location, start, end)
data = data.fetch()

# Iterate over the rows of the DataFrame and print weather data
for index, row in data.iterrows():
    # Determine if the day was sunny or rainy
    weather_type = 'Sunny' if row['prcp'] <= 0.1 else 'Rainy'
    
    # Print weather data
    print(f"Date: {index.date()}, Avg Temp: {row['tavg']}, Min Temp: {row['tmin']}, Max Temp: {row['tmax']}, Precipitation: {row['prcp']}, Weather: {weather_type}")