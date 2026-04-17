import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load data
X, y = load_iris(return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Tune model
grid = GridSearchCV(model, {'max_depth': [3, 5]}, cv=3)
grid.fit(X_train, y_train)

# Print results
print("Accuracy:", grid.score(X_test, y_test))
print("Best Params:", grid.best_params_)

# Plot tree
plot_tree(grid.best_estimator_, max_depth=2, filled=True)
plt.show()
