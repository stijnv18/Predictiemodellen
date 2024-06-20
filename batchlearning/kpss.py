from statsmodels.tsa.stattools import kpss
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('merged_data.csv')

# Create 'differenced_column' by differencing 'DE_KN_residential4_grid_import'
df['differenced_column'] = df['MeanEnergyConsumption'].diff()

# Apply KPSS test on the 'differenced_column'
statistics, p_value, n_lags, critical_values = kpss(df['differenced_column'].dropna())

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(df['differenced_column'].dropna())
plt.title('Time Series Data')
plt.show()

# Print the KPSS test results
print(f'KPSS Test Statistics: {statistics}')
print(f'p-value: {p_value}')
print('Critical Values:')
for key, value in critical_values.items():
    print(f'   {key} : {value}')
    if statistics < value:
        print(f'KPSS Test Statistics is less than {key} critical value. The series is stationary around a constant.')
    else:
        print(f'KPSS Test Statistics is greater than {key} critical value. The series is not stationary.')