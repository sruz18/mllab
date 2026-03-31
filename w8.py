import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

# Split data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, Y_train)

# Predict on test data
Y_pred = model.predict(X_test)

# Output
print("X_test:", X_test)
print("Actual:", Y_test)
print("Predicted:", Y_pred)