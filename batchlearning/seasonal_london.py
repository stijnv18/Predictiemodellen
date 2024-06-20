import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load the data from a CSV file
df = pd.read_csv('merged_data.csv')

# Convert 'DateTime' to datetime and set as index
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)

# Decompose the time series
result = seasonal_decompose(df['MeanEnergyConsumption'], model='additive', period=672)

# Plot the decomposed time series
result.plot()
plt.show()