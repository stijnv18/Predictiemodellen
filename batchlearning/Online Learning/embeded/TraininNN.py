import pandas as pd
import matplotlib.pyplot as plt
from river import compose
from river import linear_model
from river import metrics
from river import preprocessing
from river import optim
from river import neural_net
import time

# Assuming df is your DataFrame and 'consumption' is your target variable
customer_36 = pd.read_csv('customer_36.csv')
test = customer_36.tail(168)
customer_36 = customer_36.head(-168)
import numpy as np
# Create a new DataFrame with all the columns except 'consumption'
X = customer_36.drop('consumption', axis=1)
y = customer_36['consumption']

# Create a pipeline with a standard scaler and a neural network model


model = compose.Pipeline(
    preprocessing.StandardScaler(),
    neural_net.MLPRegressor(
        #3.19 = MAE: 0.027252 Modified MAE: 0.027252017222437194
        
        #Best neuron numbers: (5, 27, 13)
        #Best neuron numbers: (5, 27, 13), best MAE: 0.015826061440739662
        
        #best neuron numbers: (5, 2, 9,5), best MAE: 0.01608
        
        hidden_dims=(5,27,13),
        activations=(neural_net.activations.ReLU, 
                     
                     neural_net.activations.ReLU,
                     neural_net.activations.ReLU,
                     neural_net.activations.ReLU,

                                         
                     neural_net.activations.Identity),
        optimizer=optim.SGD(0.08),
        seed=42
    )
)

# Initialize a metric and lists to store actual values and predictions
metric = metrics.MAE()
y_true = []
y_pred_list = []

# Initialize a counter for the number of instances seen
n_instances = 0

starttime = time.time()

# Iterate over the DataFrame, updating the model and the metric
for xi, yi in zip(X.itertuples(index=False), y):
    y_pred = model.predict_one(xi._asdict())  # make a prediction
    model.learn_one(xi._asdict(), yi)  # update the model

    # Only start updating the metric and storing predictions after a year's worth of values

    metric.update(yi, y_pred)  # update the metric

    # Store actual values and predictions
    y_true.append(yi)
    y_pred_list.append(y_pred)

    # Increment the counter
    n_instances += 1

stoptime = time.time()
print('Time taken:', stoptime-starttime)


#set test index to 0
test = test.reset_index(drop=True)

# Test the model on the last week of data
y_true_test = test['consumption']
y_pred_test = []
for xi in test.drop('consumption', axis=1).itertuples(index=False):
    y_pred_test.append(model.predict_one(xi._asdict()))
    
# Calculate the MAE on the test set
metric_test = metrics.MAE()
for yi, y_pred in zip(y_true_test, y_pred_test):
    metric_test.update(yi, y_pred)
print('Test MAE:', metric_test)







