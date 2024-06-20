import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

# Load the data
df = pd.read_csv('Small LCL Data\LCL-June2015v2_0.csv')

# Convert the 'KWHpHH' column to numeric, errors='coerce' turns invalid values into NaN
df['KWHhh'] = pd.to_numeric(df['KWHhh'], errors='coerce')

# Drop NaN values
df = df.dropna(subset=['KWHhh'])

# Create the Q-Q plot
qqplot(df['KWHhh'], line='s')
plt.show()