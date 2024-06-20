from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetRegressor
import torch.nn as nn
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Assuming df is your DataFrame and 'consumption' is your target variable
customer_36 = pd.read_csv('customer_36.csv')

customer_36.head()


# Load your own data
X = customer_36.drop('consumption', axis=1).values
y = customer_36['consumption'].values

# Split the data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# Create dataloaders
train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=4)
test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=4)


# Define a function to create the model
def create_model(num_units=10, num_layers=1, learning_rate=0.01):
    layers = []
    layers.append(nn.Linear(X_train.shape[1], num_units))
    layers.append(nn.ReLU())
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(num_units, num_units))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(num_units, 1))
    model = nn.Sequential(*layers)
    return model

# Create a skorch estimator for your PyTorch model
net = NeuralNetRegressor(
    module=create_model,
    criterion=nn.MSELoss,
    optimizer=torch.optim.SGD,
    iterator_train__shuffle=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'  # use GPU if available
)

# Define the hyperparameters for the grid search
params = {
    'module__num_units': [k for k in range(1, 10)],
    'module__num_layers': [1, 2, 3],
    'lr': [0.01, 0.02, 0.05],
}

# Create the grid search
gs = GridSearchCV(net, params, refit=False, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

import pandas as pd

# Perform the grid search
gs.fit(X_train, y_train)

# Convert the results to a pandas DataFrame and save to a CSV file
results = pd.DataFrame(gs.cv_results_)
results.to_csv('grid_search_results.csv', index=False)

# Print the best parameters
print(gs.best_params_)