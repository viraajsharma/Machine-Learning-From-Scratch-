import numpy as np 
import matplotlib.pyplot as plt

#Loading The Data
x_train = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]) #features
y_train = np.array([300.0,500.0,700.0,900.0,1100.0,1300.0,1500.0,1700.0,1900.0,2100.0]) #target values

#plotting the data
plt.scatter(x_train,y_train)
plt.title("Training Data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#Function to calculate cost
def cost(x_train,y_train,weights,bias):
    m = x_train.shape[0]
    cost = 0
    for i in range(m):
        f_wb = weights*x_train[i]+bias
        cost = cost + (f_wb - y_train[i])**2
    total_cost = cost / (2 * m)
    return total_cost

#Function to calculate gradient
def gradient(x_train,y_train,weights,bias):
    m = x_train.shape[0]
    dj_dw =0
    dj_db = 0
    for i in range(m):
        f_wb = weights*x_train[i]+bias

        #calculate partial derivatives
        dj_dw_i = (f_wb - y_train[i])*x_train[i]
        dj_db_i = (f_wb - y_train[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    return dj_dw/m , dj_db/m

#function to perform gradient descent
def gradient_descent(x_train,y_train,weights,bias,learning_rate,num_iters,gradient_function):
    for i in range(num_iters):
        dj_dw,dj_db = gradient_function(x_train,y_train,weights,bias)
        weights = weights - learning_rate*dj_dw
        bias = bias - learning_rate*dj_db
    
        #update weights and bias
    return weights,bias

#initial parameters
weights_init = 0
bias_init = 0
iterations = 10000
learning_rate = 1.0e-2
weights_final , bias_final = gradient_descent(x_train,y_train,weights_init,bias_init,learning_rate,iterations,gradient)
print(f"(w,b) found by gradient descent:{weights_final:8.4f}, {bias_final:8.4f} ")

#plotting the final line
plt.scatter(x_train,y_train)
plt.plot(x_train,weights_final*x_train+bias_final)
plt.title("Simple Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
    