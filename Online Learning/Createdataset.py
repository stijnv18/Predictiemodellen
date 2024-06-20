import pandas as pd
from meteostat import Point, Hourly
import holidays
import datetime
import pytz

# Load the data
df = pd.read_csv('a.csv')

# Select 'DE_KN_residential4_grid_import' and 'export' columns
df = df[['utc_timestamp', 'DE_KN_residential3_grid_import', 'DE_KN_residential3_grid_export']]

# Remove rows where 'DE_KN_residential4_grid_import' or 'export' is NaN
df = df.dropna(subset=['DE_KN_residential3_grid_import', 'DE_KN_residential3_grid_export'])

# Convert 'utc_timestamp' to datetime and set as index
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp']).dt.tz_localize(None)

df.set_index('utc_timestamp', inplace=True)

# Get the start and end dates from the dataset
start = df.index.min().floor('h')
end = df.index.max()

# Create Point for Brussels
location = Point(50.8503, 4.3517)

# Get hourly data for the specified period
weather_data = Hourly(location, start, end)
weather_data = weather_data.fetch()

# Interpolate to every 15 minutes
weather_data = weather_data.resample('15min').interpolate()

# Add a column for weather type
weather_data['weather_type'] = weather_data['prcp']
print(weather_data)
# Select only the temperature and weather type columns
weather_data = weather_data[['temp']]

# Merge the weather data with the existing DataFrame
df = df.join(weather_data)

# Create a holiday list for Belgium for the years of the dataset
be_holidays = holidays.Belgium(years=range(start.year, end.year + 1))

# Convert the dates to datetime objects at midnight in UTC
be_holidays_utc = [pd.Timestamp(datetime.datetime.combine(date, datetime.time()).replace(tzinfo=pytz.UTC)) for date,name in be_holidays.items()]

# Convert the list to a pandas DatetimeIndex
be_holidays_index = pd.DatetimeIndex(be_holidays_utc)

# Convert the DataFrame's index to UTC
#NOTE This line is no longer needed as the index is already in UTC
#df.index = df.index.tz_localize('Europe/Brussels', ambiguous='NaT',nonexistent='shift_backward').tz_convert('UTC')

# Add a holiday column to the DataFrame
df['holiday'] = df.index.normalize().isin(be_holidays_index)

# Function to get season
#To do
#convert to binary ecoding
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

# Add 'day_of_week' and 'season' columns
df['day_of_week'] = df.index.day_name()
df['season'] = df.index.month.map(get_season)


df = pd.get_dummies(df, columns=['season'])
df = pd.get_dummies(df, columns=['day_of_week'])


# List of new 'season' and 'day_of_week' columns
season_columns = ['season_Winter', 'season_Spring', 'season_Summer', 'season_Autumn']
day_columns = ['day_of_week_Monday', 'day_of_week_Tuesday', 'day_of_week_Wednesday',
               'day_of_week_Thursday', 'day_of_week_Friday', 'day_of_week_Saturday', 
               'day_of_week_Sunday']

# Convert 'season' and 'day_of_week' columns to binary encoding
for column in season_columns + day_columns:
    df[column] = df[column].astype(int)

# Convert 'holiday' column to binary encoding
df['holiday'] = df['holiday'].astype(int)

# Export to a new CSV file
df.to_csv('residential3.csv', index=True)