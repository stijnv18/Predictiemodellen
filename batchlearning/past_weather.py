# Import Meteostat library and dependencies
from datetime import datetime, timedelta
from meteostat import Point, Daily

# Set time period to the past week
end = datetime.now()
start = end - timedelta(days=7)

# Create Point for Brussels
location = Point(50.8503, 4.3517)

# Get daily data for the past week
data = Daily(location, start, end)
data = data.fetch()

# Iterate over the rows of the DataFrame and print weather data
for index, row in data.iterrows():
    # Determine if the day was sunny or rainy
    weather_type = 'Sunny' if row['prcp'] <= 0.1 else 'Rainy'
    
    # Print weather data
    print(f"Date: {index.date()}, Avg Temp: {row['tavg']}, Min Temp: {row['tmin']}, Max Temp: {row['tmax']}, Precipitation: {row['prcp']}, Weather: {weather_type}")