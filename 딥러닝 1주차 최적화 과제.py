# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# Obtain Input data
data = pd.read_csv(('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'), header=None)
X = data.iloc[:, 3]
Y = data.iloc[:, 4]
#plt.figure(1)
plt.scatter(X, Y)

#Building a model
#Initialization
a = 0
b = 0

lr = 0.001    # Learning Rate
epochs = 1000    #1000times iterations
n = float(len(X))  # number of data

plt.figure(1)

#Performing Gradient Descent
"""
for i in range(epochs):
  Y_pred = a*X + b
  D_a = (-2/n) * sum(X * (Y - Y_pred)) #Derivative of a
  D_b = (-2/n) *sum(Y - Y_pred) #Derivative of b
  a = a -lr*D_a    #Update a
  b = b - lr*D_b   #Update b

"""

#SGD

for i in range(epochs):
  Y_pred = a*X + b
  D_a = -(Y - Y_pred)*X #Derivative of a
  D_b = -(Y - Y_pred) #Derivative of b
  a = a -lr*D_a    #Update a
  b = b - lr*D_b   #Update b


  #Making current Predictions to plot
  Y_pred = a*X + b
  plt.clf()
  plt.ylim(10, 30)
  plt.scatter(X, Y)
  plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  #Regression line
  plt.pause(0.00001)

plt.show()
"""
#Momentum
m_a, m_b, gamma = 0,0,0.9

for i in range(epochs):
  Y_pred = a*X + b
  D_a = (-1/n) * sum((Y - Y_pred) * X) #Derivative of a
  D_b = (-1/n) * sum(Y - Y_pred) #Derivative of b
  m_a = gamma * m_a - lr * D_a
  m_b = gamma * m_b - lr * D_b
  a = a + m_a    #Update a
  b = b + m_b   #Update b


#RMSProp
ag_a, ag_b, alpha = 0, 0, 0.99
for i in range(epochs):
  Y_pred = a*X + b
  D_a = (-1/n) * sum((Y - Y_pred)*X) #Derivative of a
  D_b = (-1/n) * sum(Y - Y_pred) #Derivative of b
  ag_a = alpha * ag_a + (1-alpha) * (D_a**2)   #RMSProp
  ag_b = alpha * ag_b + (1-alpha) * (D_b**2)    #RMSProp
  a = a - lr * D_a/(ag_a**0.5 + 1*math.exp(1)**-6)    #Update a
  b = b - lr * D_b/(ag_b**0.5 + 1*math.exp(1)**-6)   #Update b
  
"""