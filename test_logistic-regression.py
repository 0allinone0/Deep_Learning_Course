import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv('data.csv')
print(data.head())
inputs = data[data.columns[:2]].values
target = data['class'].values

# Visualize the data
passed = (target == 1).reshape(-1, 1)
failed = (target == 0).reshape(-1, 1)


plt.figure(1)
plt.scatter(x = inputs[passed[:, 0], 0],
                y = inputs[passed[:, 0], 1],
                marker = "^",
                color = "green",
                s = 60)
plt.scatter(x = inputs[failed[:, 0], 0],
                y = inputs[failed[:, 0], 1],
                marker = "X",
                color = "red",
                s = 60)

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(["Passed", "Failed"])

import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def compute_cost(theta, x, y):
    m = len(y)
    y_pred = sigmoid(np.dot(x , theta))
    error = (y * np.log(y_pred)) + ((1 - y) * np.log(1 - y_pred))
    cost = -1 / m * sum(error)
    gradient = 1 / m * np.dot(x.transpose(), (y_pred - y))
    return cost[0] , gradient

x1 = inputs[:, 0].reshape(-1, 1)
x2 = inputs[:, 1].reshape(-1, 1)

mean_inputs = np.mean(inputs, axis=0)
std_inputs = np.std(inputs, axis=0)
inputs = (inputs - mean_inputs) / std_inputs #normalization

rows = inputs.shape[0]
cols = inputs.shape[1]

X = np.append(np.ones((rows, 1)), inputs, axis=1) # w3
y = target.reshape(rows, 1)

theta_init = np.zeros((cols + 1, 1))
cost, gradient = compute_cost(theta_init, X, y)

print("Cost at initialization", cost)
print("Gradient at initialization:", gradient)

def gradient_descent(x, y, theta, alpha, iterations):
    costs = []
    for i in range(iterations):
        cost, gradient = compute_cost(theta, x, y)
        theta -= (alpha * gradient)
        costs.append(cost)
    return theta, costs

theta, costs = gradient_descent(X, y, theta_init, 1, 5000)

def predict(theta, x):
    results = x.dot(theta)
    return results > 0
print(theta)
p = predict(theta, X)
print("Training Accuracy:", sum(p==y)[0],"%")


plt.figure(2)
u_orig = np.linspace(data.iloc[:, 0].min(), data.iloc[:, 0].max(), 100)
v_orig = np.linspace(data.iloc[:, 1].min(), data.iloc[:, 1].max(), 100)
U_orig, V_orig = np.meshgrid(u_orig, v_orig)

grid_raw = np.c_[U_orig.ravel(), V_orig.ravel()]
grid_poly = np.hstack([
    grid_raw[:, 0:1], 
    grid_raw[:, 1:2]])

grid_norm = (grid_poly - mean_inputs) / std_inputs
grid_final = np.append(np.ones((grid_norm.shape[0], 1)), grid_norm, axis=1)

Z = grid_final.dot(theta).reshape(U_orig.shape)

u_norm = (u_orig - mean_inputs[0]) / std_inputs[0]
v_norm = (v_orig - mean_inputs[1]) / std_inputs[1]

plt.figure(2)
plt.scatter(X[passed.flatten(), 1], X[passed.flatten(), 2], marker="^", color="green", label="Passed")
plt.scatter(X[failed.flatten(), 1], X[failed.flatten(), 2], marker="X", color="red", label="Failed")
plt.contour(u_norm, v_norm, Z.T, levels=[0], colors="blue", linewidths=2)
plt.xlabel("x1 (normalized)")
plt.ylabel("x2 (normalized)")
plt.title("Decision Boundary")
plt.legend()
plt.show()