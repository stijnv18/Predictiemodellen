# %%
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TiDEModel
from darts.metrics import mae
from itertools import product

# %%
# Load the data
df = pd.read_csv('D:\\bachelor\\merged_data.csv')

# %%
# Convert the 'DateTime' column to datetime
df['DateTime'] = pd.to_datetime(df['DateTime'])

# %%
# # Set the 'DateTime' column as the index
# df.set_index('DateTime', inplace=True)

# %%
# Create a TimeSeries instance
series = TimeSeries.from_dataframe(df, 'DateTime', 'MeanEnergyConsumption')

# %%
# Create a TimeSeries instance for the covariates
covar = TimeSeries.from_dataframe(df, 'DateTime', ['cloud_cover'], ['sunshine_duration'])

# %%
# Split the data into a training set and a validation set
train, val = series.split_after(pd.Timestamp('2014-02-21'))

# %% [markdown]
# ## GRID SEARCH

# %%
# Define the parameter grid
param_grid = {
    'input_chunk_length': [12, 24, 48],
    'output_chunk_length': [1, 7, 14],
    'epochs': [10, 20, 30]
}

# %%
# Initialize the best parameters and the best score
best_params = None
best_score = float('inf')

# %%
# Generate all combinations of parameters
param_combinations = list(product(*param_grid.values()))

# # %%
# # Perform the grid search
# for params in param_combinations:
#     params_dict = dict(zip(param_grid.keys(), params))
#     model = TiDEModel(input_chunk_length=params_dict['input_chunk_length'], output_chunk_length=params_dict['output_chunk_length'])
#     model.fit(train, epochs=params_dict['epochs'], verbose=False)
#     forecast = model.predict(7*24*2)  # 7 days, 24 hours per day, 2 data points per hour
#     last_week = series[-7*24*2:]  # 7 days, 24 hours per day, 2 data points per hour
#     score = mae(forecast, last_week)
#     if score < best_score:
#         best_score = score
#         best_params = params_dict
        
# # Print the best parameters and the best score
# print('Best parameters: ', best_params)
# print('Best score: ', best_score)

# %%
common_model_args = {
    "input_chunk_length": 48,  # lookback window
    "output_chunk_length": 14,  # forecast/lookahead window
    "likelihood": None,  # use a likelihood for probabilistic forecasts
    "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
}

# %%
# Create a TIDe model
model = TiDEModel(**common_model_args)

# %%
# Train the model
model.fit(train, epochs=20, verbose=True)

# %%
# Generate a 7-day forecast
forecast = model.predict(7*24*2)  # 7 days, 24 hours per day, 2 data points per hour

# %%
# Slice the actual series to the last week
last_week = series[-7*24*2:]  # 7 days, 24 hours per day, 2 data points per hour

# %%
# Calculate the MAE of the forecast
error = mae(forecast, last_week)

# %%
print('MAE: ', error)
# Plot the forecast
last_week.plot(label='actual')
forecast.plot(label='forecast', alpha=0.5)
plt.legend()

# %%
# Create a scatter plot
length = len(df['MeanEnergyConsumption'])
print(length)
print(len(forecast))

diff = length - len(forecast)
#copy df to a new dataframe that drops the first row
df_new = df[diff:]


print(len(forecast))
print(len(df_new['MeanEnergyConsumption']))

# Convert the forecast TimeSeries to a one-dimensional array
forecast_values = forecast.values().flatten()

# Create a scatter plot
plt.scatter(df_new['MeanEnergyConsumption'], forecast_values)

# Create a 45 degree line
max_value = max(max(df_new['MeanEnergyConsumption']), max(forecast))
plt.plot([0, max_value], [0, max_value], color='red')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')



