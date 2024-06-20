import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the data
df = pd.read_csv('residential4_grid_import_export_weather_fixed_timestamps.csv')

# Convert 'utc_timestamp' to datetime format if it's not already
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])

# Define the endogenous variable (the variable we want to predict)
endog = df['DE_KN_residential4_grid_import']

# Define the exogenous variables (the variables we use to predict the endogenous variable)
exog = df.drop(['utc_timestamp', 'DE_KN_residential4_grid_import', 'DE_KN_residential4_grid_export'], axis=1)

# Fit the SARIMAX model
SARIMAX_model = auto_arima(endog, exogenous=exog,
                           start_p=1, start_q=1,
                           test='adf',
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=None, D=1, trace=False,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

# Print the model summary
print(SARIMAX_model.summary())