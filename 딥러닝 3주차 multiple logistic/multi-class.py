import numpy as np
import matplotlib.pyplot as plt
import gzip
from load_data import load_data

path = '딥러닝 3주차 multiple logistic/dataset/MNIST/{}.gz'
X_train, Y_train, X_test, Y_test = load_data(path, n_label=10)

# Normalize data in [0 1]
X_trian = X_train / 255.
X_test = X_test / 255.

# we learned this parts in the last class
def sigmoid(x):
    return 1 / (1+np.exp(-x)) # basic logistic regression

def compute_cost(theta, x, y):
    m = len(y)
    y_pred = sigmoid(np.dot(x , theta))
    error = (y * np.log(y_pred + 1e-6)) + ((1 - y) * np.log(1 - y_pred + 1e-6))
    cost = -1 / m * sum(error)
    gradient = 1 / m * np.dot(x.transpose(), (y_pred - y))
    return cost[0] , gradient
######################################
arr_cost1, arr_cost2, arr_cost3, arr_cost4, arr_cost5, arr_cost6, arr_cost7, arr_cost8, arr_cost9, arr_cost10 = [], [], [], [], [], [], [], [], [], [] # we will save a history of cost to show how to change losses

N, C = X_train.shape # N -> total number of samples, C -> data dimension

theta_1 = np.random.randn(C, 1) # model parameters for class 1 
theta_2 = np.random.randn(C, 1) # model parameters for class 2
theta_3 = np.random.randn(C, 1) # model parameters for class 3
theta_4 = np.random.randn(C, 1) # model parameters for class 4
theta_5 = np.random.randn(C, 1) # model parameters for class 5
theta_6 = np.random.randn(C, 1) # model parameters for class 6
theta_7 = np.random.randn(C, 1) # model parameters for class 7
theta_8 = np.random.randn(C, 1) # model parameters for class 8
theta_9 = np.random.randn(C, 1) # model parameters for class 9
theta_10 = np.random.randn(C, 1) # model parameters for class 10


plt.figure(1)

batch_size = 128
epoch = 3
alpha = 1e-3

# Use SGD because training samples are too huge to use GD
# Train
for i in range(epoch):
    X, Y = [], []
    idx = np.arange(N) # get all samples's indexes
    np.random.shuffle(idx) # shuffle the indexes
    for id_ in idx:
        X.append(X_train[id_])
        Y.append(Y_train[id_])
        if len(X) == batch_size:
            # update parameters
            X = np.stack(X, 0)
            Y = np.stack(Y, 0)
            # update model parameters for class 1 
            cost_1, gradient_1 = compute_cost(theta_1, X, (Y==1) * 1)
            theta_1 -= (alpha * gradient_1)
            # update model parameters for class 2
            cost_2, gradient_2 = compute_cost(theta_2, X, (Y==2) * 1)
            theta_2 -= (alpha * gradient_2)
            # update model parameters for class 3 
            cost_3, gradient_3 = compute_cost(theta_3, X, (Y==3) * 1)
            theta_3 -= (alpha * gradient_3)
            # update model parameters for class 4 
            cost_4, gradient_4 = compute_cost(theta_4, X, (Y==4) * 1)
            theta_4 -= (alpha * gradient_4)
            # update model parameters for class 5 
            cost_5, gradient_5 = compute_cost(theta_5, X, (Y==5) * 1)
            theta_5 -= (alpha * gradient_5)
            # update model parameters for class 6 
            cost_6, gradient_6 = compute_cost(theta_6, X, (Y==6) * 1)
            theta_6 -= (alpha * gradient_6)
            # update model parameters for class 7 
            cost_7, gradient_7 = compute_cost(theta_7, X, (Y==7) * 1)
            theta_7 -= (alpha * gradient_7)
            # update model parameters for class 8 
            cost_8, gradient_8 = compute_cost(theta_8, X, (Y==8) * 1)
            theta_8 -= (alpha * gradient_8)
            # update model parameters for class 9
            cost_9, gradient_9 = compute_cost(theta_9, X, (Y==9) * 1)
            theta_9 -= (alpha * gradient_9)
            # update model parameters for class 10 
            cost_10, gradient_10 = compute_cost(theta_10, X, (Y==10) * 1)
            theta_10 -= (alpha * gradient_10)
            # reset for the next batch
            X, Y = [], []
            # show how to change losses
            arr_cost1.append(cost_1)
            arr_cost2.append(cost_2)
            arr_cost3.append(cost_3)
            arr_cost4.append(cost_4)
            arr_cost5.append(cost_5)
            arr_cost6.append(cost_6)
            arr_cost7.append(cost_7)
            arr_cost8.append(cost_8)
            arr_cost9.append(cost_9)
            arr_cost10.append(cost_10)

            plt.clf()
            plt.plot(arr_cost1, label='for class 1')
            plt.plot(arr_cost2, label='for class 2')
            plt.plot(arr_cost3, label='for class 3')
            plt.plot(arr_cost4, label='for class 4')
            plt.plot(arr_cost5, label='for class 5')
            plt.plot(arr_cost6, label='for class 6')
            plt.plot(arr_cost7, label='for class 7')
            plt.plot(arr_cost8, label='for class 8')
            plt.plot(arr_cost9, label='for class 9')
            plt.plot(arr_cost10, label='for class 10')
            plt.legend(loc='upper right')
            plt.pause(0.00001)

# Test
y_pred1 = sigmoid(np.dot(X_test , theta_1))
y_pred2 = sigmoid(np.dot(X_test , theta_2))
y_pred3 = sigmoid(np.dot(X_test , theta_3))
y_pred4 = sigmoid(np.dot(X_test , theta_4))
y_pred5 = sigmoid(np.dot(X_test , theta_5))
y_pred6 = sigmoid(np.dot(X_test , theta_6))
y_pred7 = sigmoid(np.dot(X_test , theta_7))
y_pred8 = sigmoid(np.dot(X_test , theta_8))
y_pred9 = sigmoid(np.dot(X_test , theta_9))
y_pred10 = sigmoid(np.dot(X_test , theta_10))
y_pred = np.concatenate([y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, y_pred7, y_pred8, y_pred9, y_pred10], -1)
y_pred = np.argmax(y_pred, -1)

print('ACC: {:.2f}'.format(np.mean(y_pred+1 == Y_test[:, 0])*100))

plt.show()
