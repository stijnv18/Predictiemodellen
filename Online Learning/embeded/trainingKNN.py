import pandas as pd
from river import compose
from river import linear_model
from river import metrics
from river import preprocessing
from river import stream
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import numpy as np
from river import evaluate
from river import optim
import datetime as dt

# Load the data
df = pd.read_csv('merged_data.csv')


# Convert the 'utc_timestamp' column to datetime
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Drop the first row
df = df.dropna()

df['day_of_week'] = df['DateTime'].dt.dayofweek
df['hour_of_day'] = df['DateTime'].dt.hour
df['month'] = df['DateTime'].dt.month



#check for break in timestamps
import pandas as pd

# Assuming df is your DataFrame and 'timestamp' is your timestamp column
df = df.sort_values('DateTime')

# Calculate the difference between current and previous timestamp
df['time_diff'] = df['DateTime'].diff()

# Define a threshold for a break, e.g., 1 hour
threshold = pd.Timedelta(hours=1)

# Check if there are any breaks
has_breaks = any(df['time_diff'] > threshold)

print(f"Data has breaks: {has_breaks}")



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


#plotting of the predictions and actual values for a week

# Define the start and end dates
start_date = pd.to_datetime('2013-01-01')
end_date = start_date + pd.DateOffset(weeks=1)
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Filter the data
week_data = df[(df['DateTime'] >= start_date) & (df['DateTime'] < end_date)]

#combine the predictions and timestamps into a dataframe
df_prediction = pd.DataFrame({'DateTime': timestamps, 'Predictions': predictions})

#convert datetime to datetime object
df_prediction['DateTime'] = pd.to_datetime(df_prediction['DateTime'])
week_predictions = df_prediction[(df_prediction['DateTime'] >= start_date) & (df_prediction['DateTime'] < end_date)]


#save the predictions to a csv file
df_prediction.to_csv(f'knn_16_predictions.csv', index=False)
print(f"Predictions saved to knn_16_predictions.csv")
