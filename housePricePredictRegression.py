import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)
n_houses = 50
data = pd.DataFrame({
    'Area': np.random.randint(50, 200, n_houses),
    'Bedrooms': np.random.randint(1, 5, n_houses),
    'Age': np.random.randint(0, 30, n_houses),
})


data['Price'] = 1000*data['Area'] + 5000*data['Bedrooms'] - 200*data['Age'] + np.random.randint(-10000,10000,n_houses)

X = data[['Area', 'Bedrooms', 'Age']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))