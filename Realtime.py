from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("traffic_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([[data['hour'], data['day'], data['weather']]])
    prediction = model.predict(features)
    return jsonify({"predicted_congestion": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
