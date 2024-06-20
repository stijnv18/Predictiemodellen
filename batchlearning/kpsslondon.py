import pandas as pd
from statsmodels.tsa.stattools import kpss

# Load the data
df = pd.read_csv('Small LCL Data\LCL-June2015v2_0.csv')

# Convert the 'KWHpHH' column to numeric, errors='coerce' turns invalid values into NaN
df['KWHhh'] = pd.to_numeric(df['KWHhh'], errors='coerce')

# Drop NaN values
df = df.dropna(subset=['KWHhh'])

# Perform the KPSS test
result = kpss(df['KWHhh'])

# Print the KPSS Statistic and p-value
print(f'KPSS Statistic: {result[0]}')
print(f'p-value: {result[1]}')