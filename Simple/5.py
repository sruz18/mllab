import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Dataset
X = pd.DataFrame({
    'area': [1000,1500,1800,1200,2000],
    'bedrooms': [2,3,4,2,5],
    'age': [5,10,8,6,12]
})

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
kmeans.fit(X_scaled)

# Results
print("Labels:", kmeans.labels_)
print("Centroids:\n", kmeans.cluster_centers_)

# Prediction (FIXED: use DataFrame with column names)
new_data = pd.DataFrame([[1600, 3, 7]], columns=['area', 'bedrooms', 'age'])
new_data_scaled = scaler.transform(new_data)

print("New data cluster:", kmeans.predict(new_data_scaled)[0])
