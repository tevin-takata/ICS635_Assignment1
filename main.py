import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Data Preprocessing
# Use sklearn train_test_split function
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Model Training
# Console input to choose classifier
print("Choose a classifier:")
print("1 - K-Nearest Neighbors (KNN)")
print("2 - Decision Tree")
print("3 - Random Forest")

choice = input("Enter the number of the classifier you want to use: ")
if choice == "1":
    # K-Nearest Neighbors model
    # Takes integer input for number of k neighbors parameter
    # Uses StandardScaler() to fit data
    n_neighbors = int(input("Enter the number of neighbors (k) for KNN: "))
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model_name = f"KNN (k={n_neighbors})"

elif choice == "2":
    # Decision Tree model
    # Takes integer input for maximum depth parameter
    max_depth = input("Enter max_depth for Decision Tree (press Enter for default): ")
    max_depth = int(max_depth) if max_depth else None
    model = DecisionTreeClassifier(max_depth=max_depth)
    model_name = f"Decision Tree (max_depth={max_depth})"

elif choice == "3":
    # Random Forest model
    # Takes integer input for number of estimators, maximum depth, and minimum sample split parameter
    n_estimators = int(input("Enter the number of trees (n_estimators) for Random Forest: "))
    max_depth = input("Enter max_depth for Random Forest (press Enter for default): ")
    max_depth = int(max_depth) if max_depth else None
    min_samples_split = input("Enter min_samples_split for Random Forest (press Enter for default): ")
    min_samples_split = int(min_samples_split) if min_samples_split else 2
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    model_name = f"Random Forest (n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split})"

else:
    # If other value is input, exit program
    print("Invalid choice. Exiting program.")
    exit()

# Fit data using chosen model
model.fit(X_train, y_train)

# Evaluation
# Use sklearn functions to evaluate models using accuracy, precision, recall, and f1-score metrics
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Model: {model_name}")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")

# Use Seaborn to display confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix: {model_name}")
plt.show()
