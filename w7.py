import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Features: [Area, Bedrooms]
X = np.array([
    [1000, 2],
    [1500, 3],
    [1800, 4],
    [2400, 4],
    [3000, 5]
])

# Target: 0 = Low Price, 1 = High Price
Y = np.array([0, 0, 1, 1, 1])

# Create model (k = 3)
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X, Y)

# Predict
new_house = np.array([[2000, 3]])
prediction = model.predict(new_house)

# Output
print("Predicted Class:", prediction[0])