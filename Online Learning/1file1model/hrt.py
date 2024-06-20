


from river import tree, metrics, evaluate, stream
from itertools import product
import pandas as pd
from multiprocessing import Pool

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
    model = tree.HoeffdingTreeRegressor(**params_dict)
    metric = metrics.MAE()

    # Define the threshold, counter and training period
    threshold = 2  # Define your own threshold here
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

    with open('model_scores.txt', 'a') as f:
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
        'grace_period': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'delta': [1e-10, 1e-7, 1e-4],
        'tau': [0.05, 0.1, 0.2],
        #'leaf_prediction': ['mean', 'model', 'adaptive'],
        'model_selector_decay': [0.9, 0.95, 0.99],
        'min_samples_split': [2, 5, 10], 
        'binary_split': [True, False],
        'max_size': [100.0, 500.0, 1000.0],
        'memory_estimate_period': [500000, 1000000, 2000000],
        'stop_mem_management': [True, False],
        'remove_poor_attrs': [True, False],
        'merit_preprune': [True, False],
        #'bootstrap_sampling': [True, False],
        #'seed': [None, 42, 123]
    }
    param_combinations = list(product(*param_grid.values()))
    best_score = float('inf')
    best_params = None
    try:
        with Pool() as pool:
            results = pool.map(train_model, [(params, param_grid, df) for params in param_combinations])
        best_score, best_params = min(results)
        print(f'Best score: {best_score}')
        print(f'Best parameters: {best_params}')
    except KeyboardInterrupt:
        print("Interrupted by user")