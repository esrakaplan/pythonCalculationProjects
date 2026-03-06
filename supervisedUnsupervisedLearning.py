import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data = {
    "age":[22,25,47,52,46,56,23,27,48,50],
    "income":[2000,2500,8000,9000,8500,9500,2100,2600,8700,9200]
}

df = pd.DataFrame(data)

# unsupervised learning ---------------------------------------
kmeans = KMeans(n_clusters=2)

df["cluster"] = kmeans.fit_predict(df[["age","income"]])

print(df)

plt.scatter(df["age"], df["income"], c=df["cluster"])
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Customer Segments")
plt.show()

# supervised learning ------------------------------------------
X = df[["age","income"]]
y = df["cluster"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = LogisticRegression()

model.fit(X_train,y_train)

print("Accuracy:",model.score(X_test,y_test))

new_customer = [[30,3000]]

prediction = model.predict(new_customer)

print("Cluster:",prediction)