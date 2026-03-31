import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset (your data)
data = pd.DataFrame({
    'Area': [1000,1500,1800,1200,2000],
    'Bedrooms': [2,3,4,2,5],
    'Age': [5,10,8,6,12],
    'Buy': [0,1,1,0,1]   # Target (0=No, 1=Yes)
})

# Features and Target
X = data[['Area','Bedrooms','Age']]
y = data['Buy']

# Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Predict test data
y_pred = model.predict(x_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))


print("Prediction (0=No, 1=Yes):", prediction[0])

# Extra outputs
print("Tree Depth:", model.get_depth())
print("Number of Leaves:", model.get_n_leaves())
