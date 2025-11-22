from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # read raw values from the form (names in templates use underscores)
        raw_reading = request.form.get('reading_score')
        raw_writing = request.form.get('writing_score')

        # basic validation / safe conversion
        try:
            reading_score = float(raw_reading)
            writing_score = float(raw_writing)
        except (TypeError, ValueError):
            # missing or invalid numeric fields: return the form with an error message
            return render_template('home.html', results='Invalid or missing reading/writing scores')

        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = reading_score,
            writing_score = writing_score
        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")

        return render_template('home.html',results=f"{float(results[0]):.2f}")
    
if __name__=="__main__":
    app.run(host="0.0.0.0", debug = True, port=5000)