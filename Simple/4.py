import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Dataset (Hours vs Marks)
X = np.array([[1], [2], [3], [4], [5]])
Y = np.array([10, 20, 30, 40, 50])

# Model with parameters
model = DecisionTreeRegressor(
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Train
model.fit(X, Y)

# Predict
pred = model.predict([[3]])
print("Prediction for 3 hours:", pred[0])

# Plot tree
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=["Hours"], filled=True)
plt.show()
