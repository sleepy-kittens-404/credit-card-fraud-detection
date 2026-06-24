import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

TEST_DATA_PATH = os.path.join('artifacts', 'test.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('prediction.html')
    
    # Load test data and pick a random row
    test_df = pd.read_csv(TEST_DATA_PATH)
    # Separate actual fraud and legit samples
    fraud_df = test_df[test_df['Class'] == 1]
    legit_df = test_df[test_df['Class'] == 0]

    # 50/50 chance of loading a fraud vs legit sample so demo is interesting
    if np.random.rand() > 0.5 and len(fraud_df) > 0:
        sample = fraud_df.sample(1).iloc[0]
    else:
        sample = legit_df.sample(1).iloc[0]

    actual_class = int(sample['Class'])
    feature_row = sample.drop('Class').to_dict()

    # Run prediction
    custom_data = CustomData(row=feature_row)
    df = custom_data.get_data_as_dataframe()
    pipeline = PredictPipeline()
    pred, pred_proba = pipeline.predict(df)

    predicted_class = int(pred[0])
    fraud_probability = round(float(pred_proba[0][1]) * 100, 2)

    # Pick a few V features to display on the card
    display_features = {
        'Amount': round(feature_row['Amount'], 2),
        'Time (s)': round(feature_row['Time'], 2),
        'V1': round(feature_row['V1'], 4),
        'V2': round(feature_row['V2'], 4),
        'V3': round(feature_row['V3'], 4),
        'V4': round(feature_row['V4'], 4),
    }

    return render_template(
        'prediction.html',
        features=display_features,
        predicted_class=predicted_class,
        fraud_probability=fraud_probability,
        actual_class=actual_class
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)