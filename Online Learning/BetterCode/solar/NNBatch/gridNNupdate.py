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
from joblib import Parallel, delayed
import csv

# Assuming df is your DataFrame and 'consumption' is your target variable
customer_36 = pd.read_csv('customer_36.csv')

# Create a new DataFrame with all the columns except 'consumption'
X = customer_36.drop('consumption', axis=1)
y = customer_36['consumption']

# Set up logging
logging.basicConfig(filename='gridsearchV4.log',level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info(f'started running grid search for customer 36')

# Define a list of neuron numbers for the grid search
neuron_numbers = [(i, j, k, p) for i in range(1, 40) for j in range(1, 40) for k in range(1, 40) for p in range(1, 40)]

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
    modified_mae=0

    for xi, yi in zip(X.itertuples(index=False), y):
        xi_dict = dict(xi._asdict())  # Convert namedtuple to dict
        model.learn_one(xi_dict, yi)

        # Only start updating the metric and storing predictions after a month's worth of values
        if n_instances > 31*24*3:
            y_pred = model.predict_one(xi_dict)
            metric.update(yi, y_pred)
            y_pred_list.append(y_pred)

            # If more than 95% of the predictions are 0, stop training and move to the next model
            if len(y_pred_list) > 7*24 and (np.array(y_pred_list) == 0).mean() > 0.95:
                logging.info(f'Stopping training for neuron numbers: {nn} due to too many zero predictions')
                #set the metric to a high value to penalize this model
                modified_mae = 1000
                break

        # Increment the counter
        n_instances += 1

    # Calculate the standard deviation of the predictions
    std_dev = np.std(y_pred_list)

    # Add a penalty to the MAE if the standard deviation is small
    if modified_mae != 1000:
        modified_mae = metric.get() + max(0, 0.1 - std_dev)

    logging.info(f'Finished training for neuron numbers: {nn}, Modified MAE: {modified_mae}')
    print(f'Finished training for neuron numbers: {nn}, Modified MAE: {modified_mae}')
    print(f'Finished training for neuron numbers: {nn}, MAE: {metric.get()}')

    # Write the results to a CSV file
    if modified_mae != 1000:
        with open('gridsearch_resultsv4.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([nn, modified_mae])
    return nn, modified_mae


# Perform grid search using joblib to parallelize
metrics_dict = dict(Parallel(n_jobs=6)(delayed(train_model)(nn) for nn in neuron_numbers))

# Print the metrics for each neuron number
for nn, metric in metrics_dict.items():
    print(f"Neuron numbers: {nn}, MAE: {metric}")

# Find the neuron numbers that resulted in the lowest MAE
best_nn = min(metrics_dict, key=metrics_dict.get)
print(f"Best neuron numbers: {best_nn}")
#log
logging.info(f'Best neuron numbers: {best_nn}, best MAE: {metrics_dict[best_nn]}')
print(f'Best neuron numbers: {best_nn}, best MAE: {metrics_dict[best_nn]}')