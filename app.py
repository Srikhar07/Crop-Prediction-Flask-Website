from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('crop_prediction.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_values = request.form.values() #returns as strings

    float_values = [] #converting into floats
    for value in input_values:
        float_values.append(float(value))

    features = np.array([float_values]) #converting into 2d array

    scaled_values = scaler.transform(features) #trandforming the features into feature scaling

    prediction = model.predict(scaled_values) #predicting the scaled values

    return render_template('result.html', predicted_result=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)