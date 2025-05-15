import pandas as pd
# Sample historical traffic data
data = {
    'Day': ['Monday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Friday'],
    'Hour': [8, 17, 9, 18, 7, 17, 20],
    'Weather': ['Clear', 'Rain', 'Clear', 'Rain', 'Clear', 'Clear', 'Rain'],
    'Past_Congestion': [3, 8, 4, 7, 2, 9, 6],  # 1-10 scale
    'Congestion_Level': ['Medium', 'High', 'Medium', 'High', 'Low', 'High', 'Medium']
}

df = pd.DataFrame(data)
print(df)