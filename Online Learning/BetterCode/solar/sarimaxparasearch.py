import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools

# Define the p, d, q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q, and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q, and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]
print("reading data")
df = pd.read_csv('wheaterVersion.csv')
print("data read")
df['datetime'] = pd.to_datetime(df['datetime'])

# Filter the DataFrame to only include rows where 'Consumption Category' is 'GG'
df = df[df['Consumption Category'] == 'GG']

# Continue with the rest of your code
start_date = df["datetime"].min()
end_date = df['datetime'].max()

X = df.drop('consumption', axis=1)
y = df['consumption']

# Get month and day of the week from the date time column
X['Month'] = X['datetime'].dt.month
X['DayOfWeek'] = X['datetime'].dt.dayofweek

X_train = X
y_train = y

# Convert the training set back to DataFrame for the model training
train_df = pd.concat([X_train, y_train], axis=1)

#creat a subset only for customer 36
customer_36 = train_df[train_df['Customer'] == 36]

#only keep datetime month, day of the week and consumption columns
customer_36 = customer_36[['datetime', 'Month', 'DayOfWeek', 'consumption','temperature_2m','precipitation','cloud_cover','sunshine_duration','cloud_cover_low','cloud_cover_mid','cloud_cover_high']]
print(len(customer_36))

#drop evry row that is not on the hour
customer_36 = customer_36[customer_36['datetime'].dt.minute == 0]

print(len(customer_36))


# Separate the target variable and exogenous variables
exog = customer_36.drop('consumption', axis=1).drop('datetime', axis=1)
endog = customer_36['consumption']

print(exog.head())

best_aic = float("inf")
best_pdq = None
best_seasonal_pdq = None
temp_model = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            print('SARIMAX{}x{}48'.format(param, param_seasonal))
            temp_model = SARIMAX(endog,
                                 order=param,
                                 seasonal_order=param_seasonal,
                                 exog=exog,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False)
            print("Fitting")
            results = temp_model.fit()

            print("SARIMAX{}x{}48 - AIC:{}".format(param, param_seasonal, results.aic))
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
                #give inbetween update on the best pqd and seasonal pqd
                print("Best SARIMAX{}x{}48 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))
        except Exception as e:
            print("Error: ", str(e))
            continue

print("Best SARIMAX{}x{}24 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))