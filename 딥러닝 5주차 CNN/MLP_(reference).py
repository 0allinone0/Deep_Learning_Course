

import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data

class perceptron(): # one layer perceptron
    def __init__(self, c_in, c_out, is_final=False):
        # c_in is the number of input neuron
        # c_out is the number of output neuron
        self.w = np.random.rand(c_in, c_out) * 1e-3
        self.b = np.zeros([1, c_out])
        self.is_final = is_final

    def forward(self, x):
        self.x = x.copy()
        # x is input of size Batch_size X c_in
        self.h = x @ self.w + self.b
        
        if self.is_final:
            # Softmax function
            exp_h = np.exp(self.h)
            y_pred = exp_h / np.sum(exp_h, -1, keepdims=True)
        else:
            # ReLU
            y_pred = np.maximum(self.h, 0)
        return y_pred
    
    def backward(self, grad, learning_rate):
        # Compute gradient
        grad_h = grad.copy()
        if not self.is_final:
            grad_h[self.h < 0] = 0              
        grad_w = self.x.T @ grad_h
        grad_next = grad_h @ self.w.T 
        # Update parameters
        self.w = self.w - learning_rate * grad_w
        self.b = self.b - learning_rate * grad_h.mean(0, keepdims=True)
        return grad_next


F = [perceptron(28*28, 256), perceptron(256, 256), perceptron(256, 10, True)]


X_train, Y_train, X_test, Y_test = load_data()

lr = 1e-4
Loss = []
batch_size = 128
N = len(X_train)
for epoch in range(1):
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
            Label = [(Y==i)*1 for i in range(10)]
            Label = np.stack(Label, -1)
            # Forward Pass
            Y_pred = X.copy()
            for p in F:
                Y_pred = p.forward(Y_pred)
            # Compute Loss (or cost)
            loss = -np.mean(np.sum(Label * np.log(Y_pred + 1e-12), axis=1))
            Loss.append(np.mean(loss))

            # Update parameters
            grad = (Y_pred - Label)
            for p in F[::-1]:
                grad = p.backward(grad, lr)
            X, Y = [], []
            
            plt.clf()
            plt.plot(Loss)
            plt.pause(0.01)


Y_pred = X_test.copy()
for p in F:
    Y_pred = p.forward(Y_pred)
Y_pred = np.argmax(Y_pred, -1)
print('ACC: {:.2f}'.format(np.mean(Y_pred == Y_test)*100))
