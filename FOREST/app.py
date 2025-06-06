from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map')
def map_page():
    return render_template('map.html')

@app.route('/manual')
def manual_page():
    return render_template('manual.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    lat = float(request.form['lat'])
    lon = float(request.form['lon'])
    vpd_avg = float(request.form['vpd_avg'])
    vpd_max = float(request.form['vpd_max'])
    vpd_min = float(request.form['vpd_min'])
    vpd_avg_1 = float(request.form['vpd_avg_1'])
    vpd_max_1 = float(request.form['vpd_max_1'])
    vpd_min_1 = float(request.form['vpd_min_1'])

    # Scale the input data
    input_data_scaled = scaler.transform([[lat, lon, vpd_avg, vpd_max, vpd_min, vpd_avg_1, vpd_max_1, vpd_min_1]])

    # Make prediction
    prediction = rf_model.predict(input_data_scaled)

    # Display result page with prediction result
    return render_template('result.html', prediction=prediction[0], result="Your result message here")

if __name__ == '__main__':
    app.run(debug=True)
