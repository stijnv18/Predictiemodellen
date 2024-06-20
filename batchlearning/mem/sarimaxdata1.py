# %%
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# %%
df = pd.read_csv('D:\\bachelor\\dataset_resampled_hourly.csv')
print(df.head())

# %%
# Convert 'utc_timestamp' to datetime format if it's not already
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])

# Convert 'utc_timestamp' to ordinal date (number of days since a certain date)
df['utc_timestamp'] = df['utc_timestamp'].apply(lambda x: x.toordinal())

# %%
# Define the endogenous variable (the variable we want to predict)
endog = df['DE_KN_residential4_grid_import']

# Define the exogenous variables (the variables we use to predict the endogenous variable)
exog = df.drop(['DE_KN_residential4_grid_import', 'DE_KN_residential4_grid_export'], axis=1)

# %%
# Define the model
model = ARIMA(endog, order=(1, 1, 1), exog=exog)

# Fit the model
model_fit = model.fit()

# %%
# # Fit the SARIMAX model
# SARIMAX_model = pm.auto_arima(endog, exogenous=exog,
#                               start_p=1, start_q=1,
#                               test='adf',
#                               max_p=3, max_q=3, m=12,
#                               start_P=0, seasonal=True,
#                               d=None, D=1, trace=False,
#                               error_action='ignore',
#                               suppress_warnings=True,
#                               stepwise=True)

# %%
def sarimax_forecast(train, test, order, seasonal_order, exog_train, exog_test, train_target):
    model = SARIMAX(train_target, order=order, seasonal_order=seasonal_order, exog=exog_train)
    model_fit = model.fit(disp=False)
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, exog=exog_test)
    return predictions


# %%
# Split the data
train = df[:-672]
test = df[-672:]
exog_train = train[['temp', 'season_Autumn', 'season_Spring', 'season_Summer', 'season_Winter']]
exog_test = test[['temp', 'season_Autumn', 'season_Spring', 'season_Summer', 'season_Winter']]

train_target = train['DE_KN_residential4_grid_import']
test_target = test['DE_KN_residential4_grid_import']

# Define the order and seasonal_order parameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)

# Call the function
predictions = sarimax_forecast(train, test, order, seasonal_order, exog_train, exog_test, train_target)
predictions.index = test_target.index  # Set the index of the predictions to match the test_target index

# %%
# And `y_test` are the true values
mae = mean_absolute_error(test['DE_KN_residential4_grid_import'], predictions)

print(f"Mean Absolute Error: {mae}")

# %%
# Plot
plt.figure(figsize=(10,6))
plt.plot(test_target.index, test_target, label='Actual')
plt.plot(test_target.index, predictions, label='Predicted')
plt.title('SARIMAX Predictions vs Actual')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.legend(loc='best')

# %%
# Fit the model
model_fit = model.fit()

# Use a method of the 'ARIMA' class
model_fit.summary()


