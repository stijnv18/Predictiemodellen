import holidays
import datetime
import pytz

# Create a holiday list for Belgium for the years 2022 to 2025
be_holidays = holidays.Belgium(years=range(2022, 2026))

# Print all the holidays
for date, name in sorted(be_holidays.items()):
    # Convert the date to a datetime object at midnight
    dt = datetime.datetime.combine(date, datetime.time())
    # Convert the datetime object to UTC
    utc_dt = dt.replace(tzinfo=pytz.UTC)
    # Convert the UTC datetime object to a timestamp string
    utc_timestamp_str = utc_dt.isoformat()
    print(utc_timestamp_str + "Z", name)
    
be_holidays = holidays.Belgium(years=range(2016, 2020 + 1))
print(be_holidays)