from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the pre-trained model using an absolute path (modify if needed)
model_path = r'C:\Users\LENOVO\Desktop\C C++\modi-country-prediction\notebooks\rf_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    trade_volume = float(request.form['trade_volume'])
    event_next_year = 1 if request.form['event_next_year'] == 'Yes' else 0
    bilateral_relation_score = float(request.form['bilateral_relation_score'])

    # Prepare input data for the model
    input_data = pd.DataFrame([[trade_volume, event_next_year, bilateral_relation_score]],
                              columns=['Trade_Value', 'Event_Next_Year', 'Bilateral_Relation_Score'])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Expanded country mapping based on your dataset
    countries = {
        0: 'USA', 1: 'Australia', 2: 'Japan', 3: 'Russia', 4: 'Saudi Arabia',
        5: 'UK', 6: 'Canada', 7: 'Germany', 8: 'China', 9: 'Brazil',
        10: 'France', 11: 'South Korea', 12: 'Italy', 13: 'South Africa', 14: 'Nepal',
        15: 'Maldives', 16: 'Sri Lanka', 17: 'Bangladesh', 18: 'Bhutan', 19: 'Singapore',
        20: 'Indonesia', 21: 'Vietnam', 22: 'Myanmar', 23: 'Iran', 24: 'UAE',
        25: 'Oman', 26: 'Qatar', 27: 'Kuwait', 28: 'Israel', 29: 'Switzerland',
        30: 'Sweden', 31: 'Norway', 32: 'Denmark', 33: 'Netherlands', 34: 'Belgium',
        35: 'Spain', 36: 'Greece', 37: 'Turkey', 38: 'Argentina', 39: 'Chile',
        40: 'Mexico', 41: 'Colombia', 42: 'Egypt', 43: 'Nigeria', 44: 'Kenya',
        45: 'Ethiopia', 46: 'Tanzania', 47: 'Zambia', 48: 'Zimbabwe'
    }

    # Convert the prediction output back to the country name
    result = countries.get(prediction, 'Unknown Country')

    return render_template('index.html', prediction_text=f'PM Narendra Modi will likely visit {result} next year.')

if __name__ == '__main__':
    app.run(debug=True)
