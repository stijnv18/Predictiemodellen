import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
# Load the data
df = pd.read_csv('merged_data.csv')

# Convert the 'utc_timestamp' column to datetime
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Drop the first row
df = df.dropna()

df['day_of_week'] = df['DateTime'].dt.dayofweek
df['hour_of_day'] = df['DateTime'].dt.hour
df['month'] = df['DateTime'].dt.month

from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

# Prophet requires columns ds (Date) and y (value)
df = df.rename(columns={'DateTime': 'ds', 'MeanEnergyConsumption': 'y'})

# Make the prophet model and fit on the data
m = Prophet(daily_seasonality=True,yearly_seasonality=True)
m = Prophet(seasonality_mode='multiplicative')

# Tune seasonality parameters
m = Prophet(seasonality_prior_scale=0.1)

m.fit(df)

# Predict for the next year
future = m.make_future_dataframe(periods=7, freq='H')
forecast = m.predict(future)


# Perform cross-validation
# initial: the size of the initial training period
# period: the spacing between cutoff dates
# horizon: the forecast horizon
#df_cv = cross_validation(m, initial='365 days', period='2 days', horizon='7 days')

# Calculate performance metrics
#df_p = performance_metrics(df_cv)

# Plot performance metrics
#fig = plot_cross_validation_metric(df_cv, metric='mae')  # Mean Absolute Percentage Error





#fig2 = m.plot_components(forecast)

# Plot the forecast
#fig1 = m.plot(forecast)

#plt.show()