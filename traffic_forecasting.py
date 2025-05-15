import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# 1. Generate Simulated Historical and Contextual Data
def generate_data(num_samples=10000):
    np.random.seed(42)
    data = {
        'hour_of_day': np.random.randint(0, 24, size=num_samples),
        'day_of_week': np.random.randint(0, 7, size=num_samples),
        'temperature': np.random.normal(22, 5, size=num_samples),
        'precipitation': np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]),
        'event': np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1]),  # e.g., sports/concert
        'previous_congestion': np.random.uniform(0, 100, size=num_samples),
    }
    df = pd.DataFrame(data)
    # Target: current congestion level
    df['congestion_level'] = (
        0.3 * df['hour_of_day'] +
        0.2 * df['day_of_week'] +
        0.1 * df['temperature'] +
        15 * df['precipitation'] +
        20 * df['event'] +
        0.5 * df['previous_congestion'] +
        np.random.normal(0, 5, size=num_samples)
    )
    return df

# 2. Train a Predictive Model
def train_model():
    df = generate_data()
    features = ['hour_of_day', 'day_of_week', 'temperature', 'precipitation', 'event', 'previous_congestion']
    X = df[features]
    y = df['congestion_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model RMSE: {rmse:.2f}")

    joblib.dump(model, "traffic_model.pkl")
    print("Model saved as traffic_model.pkl")

if __name__ == "__main__":
    train_model()
