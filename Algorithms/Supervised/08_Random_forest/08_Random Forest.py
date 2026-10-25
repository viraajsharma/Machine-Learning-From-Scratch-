#Import libraries and load the data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#Load the Dataset
titanic_dataset = pd.read_csv('C:\\Coding\\Machine-Learning-From-Scratch-\\Algorithms\\Supervised\\08_Random_forest\\titanic.csv')
titanic_dataset = titanic_dataset.dropna(subset=["Survived"])
X = titanic_dataset[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].copy()
y = titanic_dataset["Survived"]

#Preprocess the data
X['Sex'] = X['Sex'].map({'male': 0, 'female':1})
X['Age'] = X['Age'].fillna(X['Age'].median())
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42,stratify =y)

#Train the forest
rf = RandomForestClassifier(n_estimators=100,random_state = 42)   #n_estimators=100 means the forest contains 100 trees.
rf.fit(X_train,y_train) #The model fits those trees using randomized training samples and feature subsets.

#Make predictions and evaluate the model
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
