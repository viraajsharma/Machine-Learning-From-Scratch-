import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Features: [income in thousands, age, previous purchases]
X = [
    [30, 22, 0],
    [35, 25, 0],
    [45, 28, 1],
    [60, 35, 1],
    [70, 40, 3],
    [80, 45, 2],
    [25, 50, 0],
    [90, 32, 4],
    [55, 29, 0],
    [65, 38, 2],
]

# 0 = No purchase, 1 = Purchase
y = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]

# Splitting the dataset into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

# Creating the Decision Tree Classifier
model = DecisionTreeClassifier(criterion="gini",max_depth=3,min_samples_leaf=2,random_state=42)
model.fit(X_train,y_train)

# Making predictions
predictions = model.predict(X_test)

print("Predictions:", predictions)
print("Actual:", y_test)
print("Accuracy:", accuracy_score(y_test, predictions))


#inspecting the decision tree
from sklearn.tree import export_text
print(export_text(model, feature_names=["Income", "Age", "Previous Purchases"]))