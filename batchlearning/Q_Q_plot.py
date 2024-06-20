import statsmodels.api as sm
import pandas as pd
import matplotlib.pylab as plt

df = pd.read_csv("residential4_grid_import_export_weather_fixed_timestamps.csv")
# Create 'differenced_column' by differencing 'DE_KN_residential4_grid_import'
df['differenced_column'] = df['DE_KN_residential4_grid_import'].diff()

fig, ax = plt.subplots()

sm.qqplot(df['differenced_column'], line ='45', ax = ax)
ax.set_ylim([0, 2.5])
ax.set_xlim([0, 2.5])

plt.show()