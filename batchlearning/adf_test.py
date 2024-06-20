import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Load the data
df = pd.read_csv('merged_data.csv')

# Convert the 'KWHpHH' column to numeric, errors='coerce' turns invalid values into NaN
df['MeanEnergyConsumption'] = pd.to_numeric(df['MeanEnergyConsumption'], errors='coerce')

# Drop NaN values
df = df.dropna(subset=['MeanEnergyConsumption'])

# Perform the ADF test
result = adfuller(df['MeanEnergyConsumption'])

# Print the ADF Statistic and p-value
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')