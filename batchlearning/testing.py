import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

df = pd.read_csv('D:\\bachelor\\dataset_resampled.csv')
# Convert 'utc_timestamp' to datetime format if it's not already
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])

# Set 'utc_timestamp' as the index
df = df.set_index('utc_timestamp')

# Split the data into training and test sets
train = df.loc[df.index < '2018-01-31']
test = df.loc[(df.index >= '2018-01-31') & (df.index < '2018-02-4')]

# Define the endogenous variable for the training and test sets
endog_train = train['DE_KN_residential4_grid_import']
endog_test = test['DE_KN_residential4_grid_import']

# Define the exogenous variables for the training and test sets
exog_train = train.drop('DE_KN_residential4_grid_import', axis=1)
exog_test = test.drop('DE_KN_residential4_grid_import', axis=1)
print(exog_test.shape)
# Define the order and seasonal_order parameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)

# Create the SARIMAX model
model = SARIMAX(endog_train, order=order, seasonal_order=seasonal_order, exog=exog_train)

# Fit the model
model_fit = model.fit(disp=False)

# Make predictions for the test set
predictions = model_fit.predict(start='2018-01-31', end='2018-02-04', exog=exog_test)

# Plot the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(endog_test, label='Actual')
plt.plot(predictions, color='red', label='Predicted')
plt.legend()
plt.show()