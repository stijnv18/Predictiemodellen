import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools

# Define the p, d, q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q, and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q, and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]

# Specify the data
data = pd.read_csv('merged_data.csv')

data = data.apply(pd.to_numeric, errors='coerce')

data = data['MeanEnergyConsumption']

best_aic = float("inf")
best_pdq = None
best_seasonal_pdq = None
temp_model = None

for param in pdq:
	for param_seasonal in seasonal_pdq:
		
		try:
			print('SARIMAX{}x{}24'.format(param, param_seasonal))
			temp_model = SARIMAX(data,
								order = param,
								seasonal_order = param_seasonal,
								enforce_stationarity=False,
								enforce_invertibility=False)
			print("Fitting")
			results = temp_model.fit()

			print("SARIMAX{}x{}24 - AIC:{}".format(param, param_seasonal, results.aic))
			if results.aic < best_aic:
				best_aic = results.aic
				best_pdq = param
				best_seasonal_pdq = param_seasonal
		except Exception as e:
			print("Error: ", str(e))
			continue

print("Best SARIMAX{}x{}24 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))