import pandas as pd
from river import compose
from river import metrics
from river import preprocessing
from river import stream
import matplotlib.pyplot as plt
from river import evaluate
import datetime as dt
import psutil
import os
import time
#set start time
start_time = time.time()

def print_usage():
    process = psutil.Process(os.getpid())
    print("Memory usage:", process.memory_info().rss/(1024*1024))  # in bytes 
    print("CPU usage:", process.cpu_percent(interval=1))  # interval specifies the amount of time to wait between each collection of CPU usage info


# Load the data
df = pd.read_csv('C:\\Users\\stijn\\OneDrive - Thomas More\\Bureaublad\\jaar3\\bacholororororororcode\\BetterCode\\Aicode\\merged_data.csv')
print_usage()

# Convert the 'utc_timestamp' column to datetime
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Drop the first row
df = df.dropna()

df['day_of_week'] = df['DateTime'].dt.dayofweek
df['hour_of_day'] = df['DateTime'].dt.hour
df['month'] = df['DateTime'].dt.month


stream = iter(df.itertuples(index=False))
stream = iter([(x._asdict(), y) for x, y in zip(df.drop('MeanEnergyConsumption', axis=1).itertuples(index=False), df['MeanEnergyConsumption'])])
print(next(stream))


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from river import neural_net
from river import neighbors

# Create a pipeline that includes scaling the target variable
model = compose.Pipeline(
    compose.Select('hour_of_day', 'month', 'day_of_week','temperature_2m', 'precipitation','cloud_cover','cloud_cover_low','cloud_cover_mid','cloud_cover_high','is_day','sunshine_duration'),
    preprocessing.StandardScaler(),
    preprocessing.TargetStandardScaler(
        regressor=neighbors.KNNRegressor(
            n_neighbors=16,
            aggregation_method = 'weighted_mean',
        )
    )
)
print_usage()
# Evaluate the model
steps = evaluate.iter_progressive_val_score(
    dataset=stream,
    model=model,
    metric=metrics.RMSE(),
    moment='DateTime',
    delay=dt.timedelta(days=7),
    step=1,
    yield_predictions=True,
)
print_usage()
start_date =  df['DateTime'].min()
end_date =  df['DateTime'].max()
# Initialize lists to store the predictions and their timestamps
predictions = []
timestamps = pd.date_range(start=start_date, end=end_date, freq='1h')
#remove the first timestamp
timestamps = timestamps[1:]
Mea_overtime = []
print(steps)
length = 0

for step in steps:
    
    length += 1
    predictions.append(step["Prediction"])
    Mea_overtime.append(str(step["RMSE"]).split(" ")[1])
print(step)

#convert mea_overtime to a list of floats
Mea_overtime = [float(i) for i in Mea_overtime]
print_usage()
print(Mea_overtime)

#get end time
end_time = time.time()
print(f"Runtime of the program is {end_time - start_time}")
