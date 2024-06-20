# %%
import pandas as pd
from darts import TimeSeries
from darts.models import TFTModel
from darts.metrics import mae
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from itertools import product
import matplotlib.pyplot as plt
import torch
print(torch.cuda.is_available())

# %%
# Load your data into a DataFrame
data = pd.read_csv('D:\\bachelor\\merged_data.csv')
data['DateTime'] = pd.to_datetime(data['DateTime'])

# %%
# Convert the DataFrame to a TimeSeries instance
series = TimeSeries.from_dataframe(data, 'DateTime', 'MeanEnergyConsumption')

# %%
# Split the data into a training set and a validation set
train, val = series.split_before(pd.Timestamp('2014-02-21'))

# %%
# Normalize the time series
transformer = Scaler()
train_transformed = transformer.fit_transform(train)
val_transformed = transformer.transform(val)

# %%
# Create covariates
covariates = datetime_attribute_timeseries(series, attribute='month', one_hot=True)
covariates = covariates.stack(datetime_attribute_timeseries(series, attribute='day', one_hot=True))

# %%
# Normalize the covariates
transformer_cov = Scaler()
covariates_transformed = transformer_cov.fit_transform(covariates)

# %%
# Define the parameter grid
param_grid = {
    'input_chunk_length': [12, 24, 48],
    'output_chunk_length': [1, 7, 14],
    'hidden_size': [16, 32, 64]
}

# Initialize the best parameters and the best score
best_params = None
best_score = float('inf')

# Generate all combinations of parameters
param_combinations = list(product(*param_grid.values()))

# %%
# # Perform the grid search
# for params in param_combinations:
#     params_dict = dict(zip(param_grid.keys(), params))
#     model = TFTModel(n_epochs=10, **params_dict)
#     model.fit(train_transformed, future_covariates=covariates_transformed, verbose=True)
#     forecast = model.predict(n=7, future_covariates=covariates_transformed)
#     score = mae(forecast, val_transformed)
#     if score < best_score:
#         best_score = score
#         best_params = params_dict

# # Print the best parameters and the best score
# print('Best parameters: ', best_params)
# print('Best score: ', best_score)

# %%
# print(best_params)

# Create and train the TFT model
model = TFTModel(input_chunk_length=12, output_chunk_length=7, n_epochs=10, hidden_size=16)
model.fit(train_transformed, future_covariates=covariates_transformed, verbose=True)

# %%
# Generate a forecast
forecast = model.predict(n=48*7, future_covariates=covariates_transformed)

# %%
# # Convert the normalized data back to the original scale
# original_unscaled = transformer.inverse_transform(train_transformed)

# Slice the original data to include only the time period covered by the forecast
original_slice = series.slice(forecast.start_time(), forecast.end_time())

# Inverse transform the forecast
forecast_unscaled = transformer.inverse_transform(forecast)

# %%
# Plot the forecast
original_slice.plot(label='Original')
forecast_unscaled.plot(label='Forecast', alpha=0.3, color='red')
plt.legend()

# %%
# Create a scatter plot
length = len(series['MeanEnergyConsumption'])
print(length)
print(len(forecast_unscaled))

diff = length - len(forecast_unscaled)
#copy df to a new dataframe that drops the first row
df_new = series[diff:]

# Convert the TimeSeries to a DataFrame
forecast_df = forecast_unscaled.pd_dataframe()

# Convert the DataFrame to a 1-dimensional sequence
forecast_values = forecast_df.values.flatten()

# Convert 'MeanEnergyConsumption' to a 1-dimensional sequence
actual_values = df_new.values().flatten()

#print mae
print(mae(forecast_unscaled, df_new))

plt.scatter(actual_values, forecast_values)

# Create a 45 degree line
max_value = max(max(actual_values), max(forecast_values))
plt.plot([0, max_value], [0, max_value], color='red')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')



