import numpy as np
from collections import Counter  #Counter counts occurrences.

def euclidean_distance(point1,point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def knn(training_data, training_labels, test_point, k):
    distances =[]
    for i in range(len(training_data)):
        distance = euclidean_distance(training_data[i],test_point)
        distances.append((distance,training_labels[i]))
    distances.sort(key=lambda x: x[0]) #Sort by distance
    k_nearest_labels = [label for _, label in distances[:k]]
    counter = Counter(k_nearest_labels).most_common(1)[0][0]
    return counter

training_data = [
    [1, 2],
    [2, 3],
    [3, 4],
    [6, 7],
    [7, 8]
]
training_labels = [
    'A', 'A', 'A', 'B', 'B'
]
test_point = [4,5]

k= 3
prediction = knn(training_data, training_labels, test_point, k)
print(prediction)

#Practical scikit-learn implementation of KNN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create synthetic 2D data

x,y = make_moons(n_samples=300,noise =0.3,random_state=42)
# Create a DataFrame for plotting
df = pd.DataFrame(x,columns=['Feature1','Feature2'])
df['Target'] = y
sns.scatterplot(data=df,x='Feature1',y='Feature2',hue='Target',palette='Set1')
plt.title("2D Classification Data (make_moons)")
plt.grid(True)
plt.show()

# Train-Test Split and Normalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size =0.25,random_state=42,stratify=y) #maintain roughly the same class proportions.

#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn_model=KNeighborsClassifier(n_neighbors=5) #k=5
knn_model.fit(x_train,y_train)
y_pred = knn_model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy (k=5): ",acc)

#Testing values of K
from sklearn.model_selection import cross_val_score
k_range = range(1, 21)
cv_scores = []
for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model,x_scaled,y,cv=5,scoring='accuracy')
    cv_scores.append(scores.mean())

plt.plot(
    k_range,
    cv_scores,
    marker='o'
)
plt.xlabel('k')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN: Varying Number of Neighbors')
plt.grid(True)
plt.show()
best_k = k_range[np.argmax(cv_scores)]
print(f"Best k: {best_k}")

# Train final model with best k
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(x_train, y_train)

# Predict on test data
y_pred = best_knn.predict(x_test)