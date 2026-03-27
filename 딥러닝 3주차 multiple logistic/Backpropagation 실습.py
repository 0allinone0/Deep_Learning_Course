import numpy as np
import matplotlib.pyplot as plt

# create random input and output data
x = np.array([0.05, 0.10]).reshape(2, 1)
y = np.array([0.01, 0.99]).reshape(2, 1)

w1 = np.array([[0.15, 0.20], [0.25, 0.30]])
w2 = np.array([[0.40, 0.45], [0.50, 0.55]])
w3 = np.array([[0.2, 0.12], [0.5,0.7]])


b1 = np.array([1., 0.1]).reshape(2, 1)
b2 = np.array([0.3, 0.3]).reshape(2, 1)
b3 = np.array([3., 4.]).reshape(2, 1)

learning_rate = 0.1
Loss = []

for t in range(500):
    # feed forward
    h = w1 @ x + b1
    h_relu = np.maximum(h, 0)
    h2 = w2 @ h_relu + b2
    h2_relu = np.maximum(h, 0)
    y_pred = w3 @ h_relu + b3

    # compute loss
    loss = (y_pred - y) ** 2 / 2
    Loss.append(np.sum(loss))

    # BackProp to compute a gradient of each parameter
    # 세번째 weight, bias 역전파
    grad_y_pred = (y_pred - y)
    grad_bias3 = grad_y_pred    #bias를 미분하면 1이기 때문에 그대로 
    grad_w3 = grad_y_pred @ h2_relu.T   #노트에서는 h_relu를 y_1으로 표기함

    #두번째 weight, bias backpropagation
    grad_h2_relu = w3.T @ grad_y_pred     #(y_pred - y)에 W2 곱하기 식은 노트에 나와있음  @을 사용하여 더하기 함
    grad_h2 = grad_h2_relu.copy()             
    grad_h2[h2 < 0] = 0               #활성함수 미분 0보다 작으면 0, 크면 값 유지
    grad_bias2 = grad_h2
    grad_w2 = grad_h2 @ h_relu.T

    #첫번째 weight, bias backpropagation
    grad_h1_relu = w2.T @ grad_y_pred
    grad_h1 = grad_h1_relu.copy()
    grad_h1[h < 0] = 0
    grad_bias1 = grad_h1
    grad_w1 = grad_h1 @ x.T

    #Update
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    w3 -= learning_rate * grad_w3

    b1 -= learning_rate * grad_bias1
    b2 -= learning_rate * grad_bias2
    b3 -= learning_rate * grad_bias3

print(y_pred)

plt.plot(Loss)
plt.show()