import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV , train_test_split
from sklearn.metrics import mean_squared_error , r2_score

np.random.seed(0)
x = np.random.randn(200, 6) #200 samples, 6 features
true_coef = np.array([3.2, -1.5, 0.7, 0, 2.8, -0.5])
y= x.dot(true_coef) + np.random.randn(200)*0.6 #Adding noise to the target variable

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.25,random_state=42)
ridge_model = Ridge(alpha = 1.0)
ridge_model.fit(x_train,y_train)
ridge_model.coef_  #Coefficients of the model
#Prediction
y_pred = ridge_model.predict(x_test)
mse_basic = mean_squared_error(y_test, y_pred)
print(mse_basic)
print("Coefficients (alpha = 1.0):", ridge_model.coef_)
print("R2 Score (alpha = 1.0):", r2_score(y_test, y_pred))

#Grid Search for Hyperparameter Tuning
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 500]}
grid = GridSearchCV(Ridge(),param_grid,cv=5,scoring='neg_mean_squared_error')
grid.fit(x_train, y_train)
best_ridge = grid.best_estimator_
pred_best = best_ridge.predict(x_test)
grid.best_params_["alpha"]
mse_best = mean_squared_error(y_test, pred_best)
r2_best = r2_score(y_test, pred_best)
print("Best alpha:", grid.best_params_["alpha"])
print("Mean Squared Error (Best Model):", mse_best)
print("R2 Score (Best Model):", r2_best)