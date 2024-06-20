import pandas as pd
from darts import TimeSeries
from darts.models import TFTModel
from darts.metrics import mae
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from itertools import product
import torch
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import logging
import time
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load your data into a DataFrame
data = pd.read_csv('merged_data.csv')
data['DateTime'] = pd.to_datetime(data['DateTime'])

series = TimeSeries.from_dataframe(data, 'DateTime', 'MeanEnergyConsumption')

# Split the data into a training set and a validation set
train, val = series.split_before(pd.Timestamp('2014-02-21'))


# Normalize the time series
transformer = Scaler()
train_transformed = transformer.fit_transform(train)
val_transformed = transformer.transform(val)

covariates = datetime_attribute_timeseries(series, attribute='month', one_hot=True)
covariates = covariates.stack(datetime_attribute_timeseries(series, attribute='day', one_hot=True))


transformer_cov = Scaler()
covariates_transformed = transformer_cov.fit_transform(covariates)


# Define the parameter grid
param_grid = {
    'input_chunk_length': [12, 24, 48],
    'output_chunk_length': [1, 7],
    'hidden_size': [16]
}

# Initialize the best parameters and the best score
best_params = None
best_score = float('inf')

# Generate all combinations of parameters
param_combinations = list(product(*param_grid.values()))


def evaluate_params(params):
    logger.info(f"Running with params: {params}")
    params_dict = dict(zip(param_grid.keys(), params))
    try:
        model = TFTModel(n_epochs=1, **params_dict)
        model.fit(train_transformed, future_covariates=covariates_transformed, verbose=True)
        forecast = model.predict(n=7, future_covariates=covariates_transformed)
        score = mae(forecast, val_transformed)
        logger.info(f"Finished with params: {params}. Score: {score}")
        return score, params_dict
    except Exception as e:
        logger.error(f"Error occurred with params: {params}. Error: {e}")
        return float('inf'), params_dict

def main():
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        results = list(executor.map(evaluate_params, param_combinations))

    best_score, best_params = min(results, key=lambda x: x[0])
    logger.info(f'Best parameters: {best_params}')
    logger.info(f'Best score: {best_score}')

    model = TFTModel(n_epochs=1, **best_params)
    model.fit(train_transformed, future_covariates=covariates_transformed, verbose=True)
    model.save('best_model')
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f'Total time taken: {elapsed_time} seconds')
    #save log
    with open('tftdarts.log', 'w') as f:
        f.write(f'Total time taken: {elapsed_time} seconds\n')
        f.write(f'Best parameters: {best_params}\n')
        f.write(f'Best score: {best_score}\n')
        f.write(f'Finished at: {time.ctime()}\n')

if __name__ == '__main__':
    main()




