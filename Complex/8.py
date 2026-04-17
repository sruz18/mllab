import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Dataset
# -----------------------------
data = pd.DataFrame({
    'Age': [45, 54, 60, 48, 52, 46],
    'Sex': [1, 0, 1, 0, 1, 0],
    'ChestPain': [3, 2, 1, 2, 3, 1],
    'BP': [130, 140, 120, 135, 150, 128],
    'Cholesterol': [230, 250, 240, 220, 260, 210],
    'MaxHR': [150, 140, 130, 160, 120, 170],
    'Target': [1, 1, 0, 1, 0, 1]
})

# -----------------------------
# Split features and target
# -----------------------------
X = data.drop('Target', axis=1)
y = data['Target']

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# -----------------------------
# Decision Tree
# -----------------------------
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_acc = dt.score(X_test, y_test)

# -----------------------------
# SVM
# -----------------------------
svm = SVC()
svm.fit(X_train, y_train)
svm_acc = svm.score(X_test, y_test)

# -----------------------------
# Random Forest
# -----------------------------
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test, y_test)

# -----------------------------
# Output Results
# -----------------------------
print("Decision Tree Accuracy:", dt_acc)
print("SVM Accuracy:", svm_acc)
print("Random Forest Accuracy:", rf_acc)

print("\nBest Model:")
if dt_acc > svm_acc and dt_acc > rf_acc:
    print("Decision Tree")
elif svm_acc > rf_acc:
    print("SVM")
else:
    print("Random Forest")
