from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load pre-trained model (assuming a classifier like Logistic Regression)
model = joblib.load("model.pkl")

app = Flask(__name__)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expecting JSON data
    features = np.array(data['features']).reshape(1, -1)  # Assuming input is a feature array
    prediction = model.predict(features)[0]
    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
