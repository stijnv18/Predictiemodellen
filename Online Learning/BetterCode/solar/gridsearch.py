import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
from river import compose
from river import linear_model
from river import metrics
from river import preprocessing
from river import optim
from river import neural_net
import logging
import numpy as np
# Assuming df is your DataFrame and 'consumption' is your target variable
customer_36 = pd.read_csv('customer_36.csv')




# Create a new DataFrame with all the columns except 'consumption'
X = customer_36.drop('consumption', axis=1)
y = customer_36['consumption']




# Set up logging
logging.basicConfig(filename='gridsearchV2.log',level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info(f'started running grid search for customer 36')

#create log file


# Define a list of neuron numbers for the grid search
neuron_numbers = [(i, j, k, p) for i in range(1, 30) for j in range(1, 30) for k in range(1, 30) for p in range(1, 30)]
from joblib import Parallel, delayed

# Define a function to train a model and return the MAE
import csv

# Define a function to train a model and return the MAE
def train_model(nn):
    logging.info(f'Starting training for neuron numbers: {nn}')
    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        neural_net.MLPRegressor(
            hidden_dims=nn,
            activations=(neural_net.activations.ReLU, 
                         
                         neural_net.activations.ReLU, 
                         neural_net.activations.ReLU, 
                         neural_net.activations.ReLU, 
                         neural_net.activations.ReLU, 
                         
                         neural_net.activations.Identity),
            optimizer=optim.SGD(0.08),
            seed=42
        )
    )

    metric = metrics.MAE()
    y_pred_list = []

    # Initialize a counter for the number of instances seen
    n_instances = 0

    for xi, yi in zip(X.itertuples(index=False), y):
        xi_dict = dict(xi._asdict())  # Convert namedtuple to dict
        y_pred = model.predict_one(xi_dict)
        model.learn_one(xi_dict, yi)

        # Only start updating the metric and storing predictions after a year's worth of values
        if n_instances > 365*24:
            metric.update(yi, y_pred)
            y_pred_list.append(y_pred)

        # Increment the counter
        n_instances += 1

    # Calculate the standard deviation of the predictions
    std_dev = np.std(y_pred_list)

    # Add a penalty to the MAE if the standard deviation is small
    modified_mae = metric.get() + max(0, 0.1 - std_dev)

    logging.info(f'Finished training for neuron numbers: {nn}, Modified MAE: {modified_mae}')
    print(f'Finished training for neuron numbers: {nn}, Modified MAE: {modified_mae}')
    print(f'Finished training for neuron numbers: {nn}, MAE: {metric.get()}')

    # Write the results to a CSV file
    with open('gridsearch_resultsv2.cs', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([nn, modified_mae])

    return nn, modified_mae

# Perform grid search using joblib to parallelize
metrics_dict = dict(Parallel(n_jobs=16)(delayed(train_model)(nn) for nn in neuron_numbers))

# Print the metrics for each neuron number
for nn, metric in metrics_dict.items():
    print(f"Neuron numbers: {nn}, MAE: {metric}")

# Find the neuron numbers that resulted in the lowest MAE
best_nn = min(metrics_dict, key=metrics_dict.get)
print(f"Best neuron numbers: {best_nn}")
#log
logging.info(f'Best neuron numbers: {best_nn}, best MAE: {metrics_dict[best_nn]}')
print(f'Best neuron numbers: {best_nn}, best MAE: {metrics_dict[best_nn]}')