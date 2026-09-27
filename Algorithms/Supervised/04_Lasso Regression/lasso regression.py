import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression  #This imports a function that generates artificial regression data.
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

# Generate synthetic regression data
x,y = make_regression(n_samples=500,n_features=10,n_informative=5,noise=15,random_state=42)

# Split the data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Create a Lasso regression model
Lasso_model = Lasso(alpha=0.1)  # alpha is the regularization parameter
Lasso_model.fit(x_train,y_train) #Learn the coefficients using the training data while applying the L1 penalty

#Predictions
y_pred = Lasso_model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)  
r2 = r2_score(y_test, y_pred) #R² tells you how much of the variance in the target is explained by the model relative to the baseline defined by the metric.
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # Reference line for perfect predictions
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Lasso Regression Predictions\nMSE: {mse:.2f}, R²: {r2:.2f}')
plt.show()

# The coefficient plot
plt.bar(
    range(len(Lasso_model.coef_)),
    Lasso_model.coef_
)
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.title('Lasso Regression Coefficients')
plt.show()


#Lasso Path
alphas =np.logspace(-2,1,20)  #This generates 20 values between 10^-2 and 10^1 on a logarithmic scale.
coefficients = []
for a in alphas:
    Lasso_model = Lasso(alpha=a)
    Lasso_model.fit(x_train,y_train)
    coefficients.append(Lasso_model.coef_)
coefficients = np.array(coefficients)

for i in range(coefficients.shape[1]):
    plt.plot(alphas, coefficients[:, i])
plt.xscale("log")
plt.xlabel("Alpha (λ)")
plt.ylabel("Coefficient Value")
plt.title("Lasso Path: Coefficients vs Regularization Strength")
plt.show()


