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




# Load the data
df = pd.read_csv('C:\\Users\\stijn\\OneDrive - Thomas More\\Bureaublad\\jaar3\\bacholororororororcode\\BetterCode\\Aicode\\merged_data.csv')



# Convert the 'utc_timestamp' column to datetime
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Drop the first row
df = df.dropna()

df['day_of_week'] = df['DateTime'].dt.dayofweek
df['hour_of_day'] = df['DateTime'].dt.hour
df['month'] = df['DateTime'].dt.month

df

#plot the data
plt.plot(df['DateTime'], df['MeanEnergyConsumption'])



stream = iter(df.itertuples(index=False))
stream = iter([(x._asdict(), y) for x, y in zip(df.drop('MeanEnergyConsumption', axis=1).itertuples(index=False), df['MeanEnergyConsumption'])])
print(next(stream))



from river import datasets
from river import metrics
from river import time_series

model = time_series.HoltWinters(
    alpha=0.3,
    beta=0.1,
    gamma=0.5,
    seasonality=24,
    multiplicative=True,
)

metric = metrics.MAE()

# Create a list to store the predictions
predictions = []

# Assuming 'df' is your DataFrame and 'MeanEnergyConsumption' is what you want to predict
for i, (_, row) in enumerate(df.iterrows()):
    
    y = row['MeanEnergyConsumption']
    model.learn_one(y)

    # Predict the next point only after the model has been trained on 'seasonality' number of data points
    if i >= model.seasonality:
        prediction = model.forecast(horizon=24)[0]
        predictions.append(prediction)  # Store the prediction

        # Update the metric
        metric.update(y, prediction)

print(f"Final error: {metric.get()}")
print(f"Predictions: {predictions}")

#get end time
end_time = time.time()
print(f"Runtime of the program is {end_time - start_time}")