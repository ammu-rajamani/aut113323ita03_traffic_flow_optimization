import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from flask import Flask, request, jsonify
import threading
import random
import requests
import time

# 1. Simulate GPS/IoT traffic data
def generate_gps_data(num_records=10000):
    data = {
        'vehicle_speed': np.random.normal(40, 10, num_records),
        'road_occupancy': np.random.uniform(0, 1, num_records),
        'weather_condition': np.random.randint(0, 3, num_records)
    }
    df = pd.DataFrame(data)
    df['congestion_level'] = 100 - df['vehicle_speed'] * df['road_occupancy']
    return df

# 2. Train AI Model
data = generate_gps_data()
X = data.drop(columns=['congestion_level'])
y = data['congestion_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("✅ Model Training Completed.")
print("📊 Mean Absolute Error (MAE):", round(mean_absolute_error(y_test, preds), 4))

# 3. Flask API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input_json = request.get_json()
    df = pd.DataFrame([input_json])
    prediction = model.predict(df)[0]
    return jsonify({'predicted_congestion_level': round(prediction, 2)})

# 4. Run Flask app in background
def run_flask():
    app.run(port=5000, debug=False, use_reloader=False)

thread = threading.Thread(target=run_flask)
thread.daemon = True
thread.start()

# 5. Real-Time Prediction Simulation
time.sleep(2)  # Wait for Flask to initialize
print("\n🚗 Real-Time Predictions:")
for _ in range(5):
    test_data = {
        'vehicle_speed': round(random.uniform(20, 60), 2),
        'road_occupancy': round(random.uniform(0.1, 0.9), 2),
        'weather_condition': random.choice([0, 1, 2])
    }
    response = requests.post('http://127.0.0.1:5000/predict', json=test_data)
    print("📥 Input:", test_data)
    print("📤 Output:", response.json())
    time.sleep(1)
