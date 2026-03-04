import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = pd.DataFrame({
    'Feature1': [5, 2, 3, 7, 8, 1, 6, 4],
    'Feature2': [1, 4, 2, 6, 8, 1, 5, 3],
    'Label':    [0, 0, 0, 1, 1, 0, 1, 0]
})

X = data[['Feature1','Feature2']]
y = data['Label']

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
new_points = pd.DataFrame({'Feature1':[3,7],'Feature2':[2,6]})
preds = knn.predict(new_points)
print("Predicts:", preds)

