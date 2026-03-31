import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Features: [Area, Bedrooms, Age]
X = np.array([
    [1000, 2, 5],
    [1500, 3, 10],
    [1800, 4, 8],
    [1200, 2, 6],
    [2000, 5, 12]
])

# Target: House Price
y = np.array([200000, 300000, 350000, 220000, 400000])

# Model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# New prediction
prediction = model.predict([[1600, 3, 7]])

# Output
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predicted Price:", prediction[0])

# -----------------------------
# Small Graph (Actual vs Predicted)
# -----------------------------
plt.scatter(y, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()