import pandas as pd
from river import time_series
import time
from river import metrics
from datetime import datetime, timedelta
from joblib import load
from darts import TimeSeries
import numpy as np
from influxdb_client import InfluxDBClient, Point, Dialect
from influxdb_client.client.flux_table import FluxTable
from influxdb_client.client.write_api import SYNCHRONOUS
error = []
prediction_length = 168
actual_values = []
latest_prediction = []
# Connect to the InfluxDB server
host = 'http://localhost:8086'
token = "kDdfNAmJIkm4V-dFQGebl8gIwc7VZu88u3ZEdcwMMcklivQ1ouIS3VOSo6zXLs_rw6owWpKT1NlOmi5EdWqLOA=="
org = "test"
bucket = "poggers"
client = InfluxDBClient(url=host, token=token, org=org)
# Query the data from your bucket
query = """from(bucket: "poggers")
  |> range(start: 2011-11-23T09:00:00Z, stop: 2014-02-28T00:00:00Z)
  |> filter(fn: (r) => r["_measurement"] == "measurement")
  |> filter(fn: (r) => r["_field"] == "MeanEnergyConsumption")
  |> unique()
  |> yield(name: "unique")"""

tables = client.query_api().query(query, org=org)

# Extract the data from the FluxTable objects
data = []
for table in tables:
    for record in table.records:
        data.append((record.get_time(), record.get_value()))

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data, columns=['DateTime', 'MeanEnergyConsumption'])
df['DateTime'] = pd.to_datetime(df['DateTime'])

# df = pd.read_csv('merged_data.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])

X_train = df.drop('MeanEnergyConsumption', axis=1)
y_train = df['MeanEnergyConsumption']

# Get month and day of the week from the date time column
X_train['Month'] = X_train['DateTime'].dt.month
X_train['DayOfWeek'] = X_train['DateTime'].dt.dayofweek

# Convert the training set back to DataFrame for the model training
train_df = pd.concat([X_train, y_train], axis=1)

model_without_exog = (time_series.SNARIMAX(p=1,d=0,q=1,sp=0,sd=1,sq=1,m=24))

mae_without_exog = metrics.MAE()
for i, (_, row) in enumerate(train_df.iterrows()):
	y = row['MeanEnergyConsumption']
	model_without_exog.learn_one(y)
	if i > 0:  # Skip the first observation
		forecast = model_without_exog.forecast(horizon=prediction_length)  # forecast 1 step ahead
		mae_without_exog.update(y, forecast[prediction_length-1])
		actual_values.append((row['DateTime'], y))
		actual_values = actual_values[-168:]
		# Save the latest prediction
		latest_prediction.append((row['DateTime'] + timedelta(hours=prediction_length), forecast[prediction_length-1]))
		latest_prediction = latest_prediction[-168-prediction_length:]
		error.append(mae_without_exog.get())
	time.sleep(0.1)