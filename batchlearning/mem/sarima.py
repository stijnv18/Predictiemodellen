# %%
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error

# %%
# Load the CSV data into a pandas DataFrame
df = pd.read_csv('D:\\bachelor\\merged_data.csv')
# Display the first few rows of the DataFrame
print(df.head())

# %%
# Assume 'date_time' is the column in your DataFrame that contains the date and time information
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Set 'date_time' as the index of the DataFrame
df.set_index('DateTime', inplace=True)

df = df[:-48*14]

# %%
print(df.head())
print(df.columns)

# %%
# Now 'time_series' has a DatetimeIndex
time_series = df.index.to_series()

# %%
# Resample the time series to an hourly frequency
time_series = time_series.resample('H').mean()

# %%
# Define the order parameters for the ARIMA model and the seasonal component
# These are just example values - you'll need to choose appropriate values based on your data
order = (1, 0, 1)
seasonal_order = (1, 1, 1, 24)

# %%
# Convert the time_series to numeric, coercing non-numeric values to NaN
time_series = pd.to_numeric(time_series, errors='coerce')

# Handle NaN values. Here, we're filling them with the mean of the other values
# You might want to handle them differently depending on your specific dataset and problem
time_series.fillna(time_series.mean(), inplace=True)

# %%
print(time_series)

# %%
# # Fit the SARIMA model
# model = sm.tsa.statespace.SARIMAX(time_series, order=order, seasonal_order=seasonal_order)
# results = model.fit()
# Fit the model
model = sm.tsa.statespace.SARIMAX(df['MeanEnergyConsumption'], order=order, seasonal_order=seasonal_order)
results = model.fit()

# %%
# Print the model summary
print(results.summary())
# Plot diagnostics
results.plot_diagnostics(figsize=(15, 12))

# %%
# Get predictions and confidence intervals
pred = results.get_prediction(start=pd.to_datetime('2014-02-14 00:00:00'), dynamic=False)
pred_conf = pred.conf_int()

# Plot the data
plt.figure(figsize=(15, 12))

# Plot the actual data
plt.plot(df['MeanEnergyConsumption'].last('7D'), label='Actual data')

# Plot the predicted data
plt.plot(pred.predicted_mean, color='red', label='Predicted data', alpha=.7)

# Plot the confidence intervals
plt.fill_between(pred_conf.index,
                 pred_conf.iloc[:, 0],
                 pred_conf.iloc[:, 1], color='pink', alpha=.3)

# Add a legend
plt.legend()