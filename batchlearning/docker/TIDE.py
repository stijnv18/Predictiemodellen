import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TiDEModel
from darts.metrics import mae
from itertools import product
import pickle
# Load the data
df = pd.read_csv('customer_36.csv')
# Convert the 'DateTime' column to datetime
df['DateTime'] = pd.to_datetime(df['DateTime'])
# Create a TimeSeries instance
series = TimeSeries.from_dataframe(df, 'DateTime', 'MeanEnergyConsumption')
# Create a TimeSeries instance for the covariates
covar = TimeSeries.from_dataframe(df, 'DateTime', ['cloud_cover'], ['sunshine_duration'])
# Split the data into a training set and a validation set
train, val = series.split_after(pd.Timestamp('2014-02-21'))
common_model_args = {
    "input_chunk_length": 48,  # lookback window
    "output_chunk_length": 14,  # forecast/lookahead window
    "likelihood": None,  # use a likelihood for probabilistic forecasts
    "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
}
# Create a TIDe model
model = TiDEModel(**common_model_args)
# Train the model
model.fit(train, epochs=20, verbose=True)
# Generate a 7-day forecast
forecast = model.predict(7*24*2)  # 7 days, 24 hours per day, 2 data points per hour
# Slice the actual series to the last week
last_week = series[-7*24*2:]  # 7 days, 24 hours per day, 2 data points per hour
# Calculate the MAE of the forecast
error = mae(forecast, last_week)
print('MAE: ', error)

# Generate a 3-day forecast
forecast = model.predict(3*24*2)  # 3 days, 24 hours per day, 2 data points per hour

# Save the model
with open('my_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Print the forecasted values
print('Forecasted values: ', forecast.values())