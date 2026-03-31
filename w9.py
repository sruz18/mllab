import pandas as pd
from sklearn.cluster import KMeans

# Dataset (your data)
data = pd.DataFrame({
    'area': [1000,1500,1800,1200,2000],
    'bedrooms': [2,3,4,2,5],
    'age': [5,10,8,6,12]
})

print(data)

# Features only (no target)
X = data[['area','bedrooms','age']]

# Model (K = 2 clusters)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Output
print("\nCluster Labels:", kmeans.labels_)
print("Centroids:\n", kmeans.cluster_centers_)

# Predict new house cluster
new_data = pd.DataFrame([[1600,3,7]], columns=['area','bedrooms','age'])
print("Cluster for new house:", kmeans.predict(new_data)[0])