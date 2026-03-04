import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


np.random.seed(42)
n_customers = 20
data = pd.DataFrame({
    'Spending_Score': np.random.randint(10, 100, n_customers),
    'Income': np.random.randint(20, 150, n_customers)
})

X = data[['Spending_Score', 'Income']]


kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)


plt.scatter(data['Income'], data['Spending_Score'], c=data['Cluster'], cmap='viridis')
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation with KMeans")
plt.show()

print("=== Customer Segments ===")
print(data)