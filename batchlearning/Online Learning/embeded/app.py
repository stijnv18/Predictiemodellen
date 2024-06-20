from flask import Flask, render_template, jsonify
from threading import Thread
import plotly
import plotly.graph_objs as go
import json
import pandas as pd
import time

from river import compose
from river import linear_model
from river import optim
from river import preprocessing
from river import time_series
from sklearn.model_selection import train_test_split
import pandas as pd
from river import neighbors
import time
from river import metrics


app = Flask(__name__)

# Global variable to store the latest prediction
latest_prediction = []

def create_graph(dates, values):
    # Generate your graph
    graph = go.Figure(
        data=[go.Scatter(x=dates, y=values)]
    )

    # Convert the figures to JSON
    graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/')
def index():
    # Fetch the latest prediction
    prediction = get_latest_prediction().get_json()

    # Check if there is a prediction
    if not prediction['dates'] or not prediction['values']:
        # No prediction available, return a default page
        return render_template('index.html', graphJSON='null')

    # Unpack the dates and values from the prediction
    dates = prediction["dates"]
    values = prediction["values"]

    # Call the function to create the graph
    graphJSON = create_graph(dates, values)

    return render_template('index.html', graphJSON=graphJSON)

@app.route('/latest_prediction', methods=['GET'])
def get_latest_prediction():
    if not latest_prediction:
        return jsonify({'dates': [], 'values': []})

    dates, values = zip(*latest_prediction)
    dates = [date.isoformat() for date in dates]
    #print("sending update", dates, values)
    return jsonify({'dates': dates, 'values': values})


@app.route('/start_training', methods=['POST'])
def start_training():
    # Start model training in a new thread
    print("Starting model training...")
    thread = Thread(target=run_model)
    thread.start()
    return jsonify({'status': 'Model training started'})


def run_model():
    global latest_prediction
    print("Running the model training...")
    df = pd.read_csv('merged_data.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    X = df.drop('MeanEnergyConsumption', axis=1)
    y = df['MeanEnergyConsumption']

    # Get month and day of the week from the date time column
    X['Month'] = X['DateTime'].dt.month
    X['DayOfWeek'] = X['DateTime'].dt.dayofweek

    X_train = X
    y_train = y

    # Convert the training set back to DataFrame for the model training
    train_df = pd.concat([X_train, y_train], axis=1)

    model_without_exog = (
    time_series.SNARIMAX(
        p=1,
        d=0,
        q=1,
        sp=0,
        sd=1,
        sq=1,
        m=24
        )
    )
    mae_without_exog = metrics.MAE()
    for i, (_, row) in enumerate(train_df.iterrows()):
        y = row['MeanEnergyConsumption']
        model_without_exog.learn_one(y)
        if i > 0:  # Skip the first observation
            forecast = model_without_exog.forecast(horizon=1)  # forecast 1 step ahead
            mae_without_exog.update(y, forecast[0])

            # Save the latest prediction
            latest_prediction.append((row['DateTime'], forecast[0]))
        time.sleep(0.1)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)