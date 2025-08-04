# train_and_save_model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

# Dummy training data
data = {
    'ssc_percentage': [80, 60, 77],
    'hsc_percentage': [78, 65, 70],
    'degree_percentage': [70, 60, 69],
    'emp_test_percentage': [85, 55, 68],
    'mba_percent': [70, 60, 68.63],
    'gender_M': [1, 0, 0],
    'ssc_board_Others': [0, 1, 0],
    'hsc_board_Others': [0, 1, 0],
    'hsc_subject_Commerce': [0, 0, 0],
    'hsc_subject_Science': [1, 1, 1],
    'undergrad_degree_Others': [0, 0, 0],
    'undergrad_degree_Sci&Tech': [1, 1, 1],
    'work_experience_Yes': [0, 1, 0],
    'specialisation_Mkt&HR': [1, 0, 0],
    'status': [1, 0, 1]
}

df = pd.DataFrame(data)
X = df.drop('status', axis=1)
y = df['status']

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save it to placement.pkl
with open('placement.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as placement.pkl")
