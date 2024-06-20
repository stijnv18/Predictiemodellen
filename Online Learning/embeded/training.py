import calendar
import math
from river import compose
from river import linear_model
from river import optim
from river import preprocessing
from river import time_series
from sklearn.model_selection import train_test_split
import pandas as pd
from river import neighbors
import time
from river import metrics

# Assuming df is your DataFrame and 'MeanEnergyConsumption' is your target variable
df = pd.read_csv('merged_data.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])

#take out a snipit of the data for testing that is one week long from the end of the data
test_data = df[df['DateTime'] > df['DateTime'].max() - pd.Timedelta(days=7)]


#get date range 
start_date = df["DateTime"].min()
end_date = df['DateTime'].max()

X = df.drop('MeanEnergyConsumption', axis=1)
y = df['MeanEnergyConsumption']

# Get month and day of the week from the date time column
X['Month'] = X['DateTime'].dt.month
X['DayOfWeek'] = X['DateTime'].dt.dayofweek


X_train = X
y_train = y

# Convert the training set back to DataFrame for the model training
train_df = pd.concat([X_train, y_train], axis=1)

#plot the data


#plot the last week of data
last_week = train_df[train_df['DateTime'] > end_date - pd.Timedelta(days=7)]


model_without_exog = (
    time_series.SNARIMAX(
        p=1,
        d=0,
        q=1,
        sp=0,
        sd=1,
        sq=1,
        m=24
    )
)

mae_without_exog = metrics.MAE()


startTime = time.time()

for i, (_, row) in enumerate(train_df.iterrows()):
    y = row['MeanEnergyConsumption']
    model_without_exog.learn_one(y)
    if i > 0:  # Skip the first observation
        forecast = model_without_exog.forecast(horizon=1)  # forecast 1 step ahead
        mae_without_exog.update(y, forecast[0])

print(f"One-step-ahead MAE without exogenous data: {mae_without_exog.get()}")

endTime = time.time()

diff = endTime-startTime

print("seconds taken to train:",diff)
