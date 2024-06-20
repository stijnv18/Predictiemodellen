

import matplotlib.pyplot as plt

from river import tree, metrics, evaluate, stream,linear_model as linear
from itertools import product
import pandas as pd
from multiprocessing import Pool
from river import compose
from river import preprocessing

def load_and_preprocess_data():
    df = pd.read_csv('residential4_grid_import_export_weather_fixed_timestamps.csv')
    df['DE_KN_residential4_grid_import'] = df['DE_KN_residential4_grid_import'].diff()
    df['DE_KN_residential4_grid_import'].iloc[0] = 0
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df['day_of_week'] = df['utc_timestamp'].dt.dayofweek
    df['hour_of_day'] = df['utc_timestamp'].dt.hour
    df['month'] = df['utc_timestamp'].dt.month
    return df
def train_model(args):
    params, param_grid, df = args
    print("testing params: ", params)
    params_dict = dict(zip(param_grid.keys(), params))
    model = linear.PARegressor(**params_dict)
    metric = metrics.MAE()



    # Define the threshold, counter and training period
    threshold = 20  # Define your own threshold here
    counter = 0
    training_period = 31*3*24  # Define your own training period here

    for i, row in enumerate(df.itertuples(index=False, name=None)):
        x = dict(zip(df.columns, row[:-1]))  # convert tuple to dictionary
        y = row[-1]  # assuming 'target' is the last column in the dataframe
        y_pred = model.predict_one(x)
        metric.update(y, y_pred)
        model.learn_one(x, y)
        score = metric.get()

        # Start checking the threshold after the training period
        if i > training_period:
            # Check if the score is above the threshold
            if score > threshold:
                counter += 1
                # If the score is above the threshold for 5 consecutive iterations, stop training
                if counter >= 24:
                    print("Stopping training due to high error.")
                    break
            else:
                counter = 0  # Reset the counter if the score is below the threshold

    with open('Parscore.txt', 'a') as f:
        f.write(f'Params: {params_dict}, Score: {score}\n')
    return score, params_dict

if __name__ == '__main__':

    df = load_and_preprocess_data()
    features = ['hour_of_day','day_of_week','month', 'temp','season_Summer','season_Winter','season_Spring','season_Autumn','holiday']
    target = 'DE_KN_residential4_grid_import'

    # New code to filter the dataframe
    df = df[features + [target]]
        
    # Define the parameter grid
    param_grid = {
        'C': [0.000_005,0.000_01,0.000_05,0.000_04,0.000_06],
        'mode': [0, 1, 2],
        'eps': [0.000_07,0.000_1,0.000_2,],
        'learn_intercept': [True, False]
    }
    param_combinations = list(product(*param_grid.values()))
    best_score = float('inf')
    best_params = None
    try:
        with Pool() as pool:
            results = pool.map(train_model, [(params, param_grid, df) for params in param_combinations])
        best_score, best_params = min(results, key=lambda x: x[0])
        print(f'Best score: {best_score}')
        print(f'Best parameters: {best_params}')

        # Retrain the model with the best parameters
        best_model = linear.PARegressor(**best_params)
        y_pred = []
        y_true = []
        for _, row in df.iterrows():
            xi = row.drop(target)
            yi = row[target]
            y_pred.append(best_model.predict_one(xi.to_dict()))
            best_model.learn_one(xi.to_dict(), yi)
            y_true.append(yi)

        # Plot the predicted values
        plt.plot(y_true,alpha=0.5 ,label='True')
        plt.plot(y_pred,alpha=0.5, label='Predicted')
        plt.legend()
        plt.show()
        

        # Scatter plot of true vs predicted values
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot(y_true, y_true, 'r')  # red line for perfect predictions
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)  # add a grid
        plt.show()
        
        plt.scatter(y_true[-168:], y_pred[-168:], alpha=0.5)
        plt.plot(y_true[-168:], y_true[-168:], 'r')  # red line for perfect predictions
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)  # add a grid
        plt.show()
        
        
        plt.plot(y_true[-168:],alpha=0.5 ,label='True')
        plt.plot(y_pred[-168:],alpha=0.5, label='Predicted')
        plt.legend()
        plt.show()

    except KeyboardInterrupt:
        print("Interrupted by user")