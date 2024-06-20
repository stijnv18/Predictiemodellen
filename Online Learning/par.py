import pandas as pd
from river import compose
from river import linear_model
from river import metrics
from river import preprocessing
from river import stream
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt

df = pd.read_csv('residential4_grid_import_export_weather_fixed_timestamps.csv')

# Convert the 'utc_timestamp' column to datetime
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])


df['day_of_week'] = df['utc_timestamp'].dt.dayofweek
df['hour_of_day'] = df['utc_timestamp'].dt.hour
#extract the season
df['month'] = df['utc_timestamp'].dt.month
print(df['hour_of_day'])

# Define the features and the target
features = ['hour_of_day', 'temp','day_of_week','season_Summer','season_Winter','month','season_Spring','season_Autumn','holiday']
target = 'DE_KN_residential4_grid_import'

# Create a feature selector
selector = SelectKBest(score_func=f_regression, k=2)

# Fit the selector to the data
selector.fit(df[features], df[target])

# Get the selected features
mask = selector.get_support()
selected_features = [f for f, m in zip(features, mask) if m]

print(f'Selected features: {selected_features}')

# Create a model
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.PARegressor()
    #linear_model.LinearRegression()
)

metric = metrics.MAE()

# Iterate over the data and update the model and the metric
for xi, yi in stream.iter_pandas(df[selected_features], df[target]):
    y_pred = model.predict_one(xi) if model else None
    model.learn_one(xi, yi)
    if y_pred is not None:
        metric.update(yi, y_pred)

print(f'MAE: {metric.get()}')

scores = selector.scores_

# Create a list of tuples (feature, score)
feature_scores = list(zip(features, scores))

# Sort the features by score
sorted_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)

# Plot the feature scores
plt.bar(*zip(*sorted_features))
plt.xticks(rotation=90)
plt.show()