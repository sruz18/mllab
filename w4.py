import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
x = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([2,4,5,4,5])

# Create model
model = LinearRegression()
model.fit(x, y)

# Predict
y_pred = model.predict(x)

# Output
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# Graph
plt.scatter(x, y)
plt.plot(x, y_pred)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simple Linear Regression")
plt.show()