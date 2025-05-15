import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Encode categorical variables
le_day = LabelEncoder()
le_weather = LabelEncoder()
le_congestion = LabelEncoder()

data = {
    'Day': ['Monday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Friday'],
    'Hour': [8, 17, 9, 18, 7, 17, 20],
    'Weather': ['Clear', 'Rain', 'Clear', 'Rain', 'Clear', 'Clear', 'Rain'],
    'Past_Congestion': [3, 8, 4, 7, 2, 9, 6],  # 1-10 scale
    'Congestion_Level': ['Medium', 'High', 'Medium', 'High', 'Low', 'High', 'Medium']
}

df = pd.DataFrame(data)
df['Day'] = le_day.fit_transform(df['Day'])
df['Weather'] = le_weather.fit_transform(df['Weather'])
df['Congestion_Level_Label'] = le_congestion.fit_transform(df['Congestion_Level'])

# Define features and labels
X = df[['Day', 'Hour', 'Weather', 'Past_Congestion']]
y = df['Congestion_Level_Label']

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict for new input
new_data = [[le_day.transform(['Monday'])[0], 8, le_weather.transform(['Clear'])[0], 6]]
predicted_label = model.predict(new_data)[0]
predicted_congestion = le_congestion.inverse_transform([predicted_label])[0]

print(f"Predicted Congestion Level: {predicted_congestion}")