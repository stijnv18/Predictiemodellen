import datetime
import pytz

# Your UTC timestamp string
utc_timestamp_str = "2022-12-25T00:00:00Z"

# Convert the string to a datetime object
dt = datetime.datetime.strptime(utc_timestamp_str, "%Y-%m-%dT%H:%M:%SZ")

# Make the datetime object timezone aware
dt = dt.replace(tzinfo=pytz.UTC)

# Get the day of the week
day_of_week = dt.strftime("%A")

# Get the season
month = dt.month
if month in [12, 1, 2]:
    season = "Winter"
elif month in [3, 4, 5]:
    season = "Spring"
elif month in [6, 7, 8]:
    season = "Summer"
else:
    season = "Autumn"

# Get the month
month = dt.strftime("%B")

# Get the day
day = dt.day

# Get the hour
hour = dt.hour

print("Day of week:", day_of_week)
print("Season:", season)
print("Month:", month)
print("Day:", day)
print("Hour:", hour)