from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("residential4_grid_import_export_weather_fixed_timestamps.csv")

# Create 'differenced_column' by differencing 'DE_KN_residential4_grid_import'
df['differenced_column'] = df['DE_KN_residential4_grid_import'].diff()

df['differenced_column'].fillna(method='backfill', inplace=True)
# Set the first value in 'differenced_column' to 0
# df['differenced_column'].iloc[0] = 0
print(df['differenced_column'])

# # Define the constant
# constant = abs(df['differenced_column'].min()) + 1

# # Add the constant to 'differenced_column'
# df['differenced_column'] = df['differenced_column'] + constant

result = seasonal_decompose(df['differenced_column'], model='additive', period=672)
trend = result.trend.dropna()
seasonal = result.seasonal.dropna()
residual = result.resid.dropna()

# Plot the decomposed components
plt.figure(figsize=(6,6))

df.index = pd.to_datetime(df['utc_timestamp'])

plt.subplot(5, 1, 1)
plt.plot(df['DE_KN_residential4_grid_import'], label='Original Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.subplot(5, 1, 2)
plt.plot(trend, label='Trend')
plt.ylabel('Value')
plt.legend()

plt.subplot(5, 1, 3)
plt.plot(seasonal, label='Seasonal')
plt.ylabel('Value')
plt.legend()

plt.subplot(5, 1, 4)
plt.plot(residual, label='Residuals')
plt.ylabel('Value')
plt.legend()

plt.subplot(5,1,5)
plt.plot(df['differenced_column'], label='Differenced Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.tight_layout()
plt.show()