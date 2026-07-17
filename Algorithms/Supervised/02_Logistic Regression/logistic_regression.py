import numpy as np
import matplotlib.pyplot as plt

#Generating the data
x_train = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]) #features
y_train = np.array([0,0,0,0,0,1,1,1,1,1]) #target values

#plottiing the data
plt.scatter(x_train,y_train)
plt.title("Training Data")
plt.xlabel("x")
plt.ylabel('y')
plt.show()


#Sigmoid Function
def sigmoid(z):
    return 1/(1+np.exp(-z))

#Cost Function
def cost(x_train,y_train,weights,bias):
    m = x_train.shape[0]
    cost = 0
    for i in range(m):
        epsilon = 1e-15
        z = np.dot(weights,x_train[i])+bias
        f_wb = sigmoid(z)

        # Prevent log(0)
        f_wb = np.clip(f_wb, epsilon, 1 - epsilon)

        cost += -y_train[i]*np.log(f_wb) - (1-y_train[i])*np.log(1-f_wb)
    total_cost = cost/m
    return total_cost

#Gradient Function
def gradient(x_train,y_train,weights,bias):
    m = x_train.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = sigmoid(np.dot(x_train[i],weights)+bias)
        err = f_wb - y_train[i]
        dj_dw += err*x_train[i]
        dj_db += err
    return dj_dw/m , dj_db/m

#Gradient Descent Function
def gradient_descent(x_train,y_train,weights,bias,learning_rate,iteration,gradient_function):
    cost_history = []
    for i in range(iteration):
        dj_dw,dj_db = gradient_function(x_train,y_train,weights,bias)
        weights = weights - learning_rate*dj_dw
        bias = bias - learning_rate*dj_db
        cost_history.append(cost(x_train,y_train,weights,bias))
    return weights,bias,cost_history

#Initial parameters
weights_init = 0
bias_init = 0
iterations = 10000
learning_rate = 1.0e-2
weights_final, bias_final,cost_history = gradient_descent(x_train,y_train,weights_init,bias_init,learning_rate,iterations,gradient)

#plotting the final line
plt.scatter(x_train,y_train)
plt.plot(x_train,sigmoid(weights_final*x_train+bias_final))
plt.title("Logistic Regression Result")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


#plotting the Loss function
plt.plot(cost_history)
plt.title("Cost vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()
