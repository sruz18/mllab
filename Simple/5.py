import pandas as pd
from sklearn.cluster import KMeans

# Dataset
X = pd.DataFrame({
    'area': [1000,1500,1800,1200,2000],
    'bedrooms': [2,3,4,2,5],
    'age': [5,10,8,6,12]
})

# Model
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Results
print("Labels:", kmeans.labels_)
print("Centroids:\n", kmeans.cluster_centers_)

# Prediction
print("New data cluster:", kmeans.predict([[1600,3,7]])[0])
