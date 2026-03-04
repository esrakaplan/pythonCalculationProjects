import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

np.random.seed(42)
n_students = 100
data = pd.DataFrame({
    'Attendance': np.random.randint(50, 100, n_students),
    'Study_Hours': np.random.randint(1, 10, n_students),
    'GPA': np.random.uniform(1, 4, n_students),
})


data['Passed'] = ((data['GPA'] > 2.5) & (data['Attendance'] > 70)).astype(int)


X = data[['Attendance', 'Study_Hours', 'GPA']]
y = data['Passed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------- Model ---------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# --------- Sonuç ---------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))