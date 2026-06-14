import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate industrial motor sensor data
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'temperature': np.random.normal(75, 15, n),
    'vibration': np.random.normal(0.5, 0.2, n),
    'current': np.random.normal(10, 2, n),
    'fault': np.random.choice([0, 1], n, p=[0.85, 0.15])
})

# Train model
X = data[['temperature', 'vibration', 'current']]
y = data['fault']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy:.2%}")
print("Motor fault detection ready!")
