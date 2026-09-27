import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures   #It doesn't perform regression. It creates new features from your existing features.
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r'C:\Users\viraa\Downloads\data.csv')  #Reading the data from csv file.

x = data.iloc[:,1:2].values   #iloc means selecting data based on integer positions.
y = data.iloc[:,2].values     # syntax = datas.iloc[row_selection, column_selection]

#Ordinary Linear Regression
line1 = LinearRegression()
line1.fit(x,y)  #Use X and y to learn the parameters of the model.


#Polynomial Regression
poly = PolynomialFeatures(degree=4, include_bias = False) #degree = 4 means we are creating new features by raising the existing features to the power of 4.
x_poly = poly.fit_transform(x)  #The transformer examines the feature structure and determines what transformation is needed and actually transforms the data.
line2 = LinearRegression()
line2.fit(x_poly,y)

#Linear Regression visualization
plt.scatter(x, y, color='blue')

plt.plot(x, line1.predict(x), color='red')
plt.title('Linear Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')

plt.show()

#Polynomial Regression visualization
plt.scatter(x,y,color = 'blue')
plt.plot(x,line2.predict(x_poly),color = 'red')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')

plt.show()