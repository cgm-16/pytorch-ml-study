# Learning about Binary Classification
# Binary Classification is a type of classification task where the goal is to predict one of two possible classes or outcomes based on input features.

# A sigmoid function is used in logistic regression to map predicted values to probabilities,
# it is important to understand how the sigmoid function works and how it can be used to make predictions.
# The sigmoid function is defined as:
# f(x) = 1 / (1 + e^(-x))
# where e is the base of the natural logarithm, and x is the input value.

# %matplotlib inline is used to display plots inline in Jupyter notebooks. (The more you know!)

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, 'g')
plt.plot([0, 0], [1, 0], ':')
plt.title('Sigmoid Function')
plt.show()

# Changing the slope of the sigmoid function
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5 * x)
y2 = sigmoid(x)
y3 = sigmoid(2 * x)

plt.plot(x, y1, 'r', linestyle='--')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'b', linestyle='--')
plt.plot([0, 0], [1, 0], ':')
plt.title('Sigmoid Function with Different Slopes')
plt.show()

# Changing the intercept of the sigmoid function
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x + 0.5)
y2 = sigmoid(x + 1)
y3 = sigmoid(x + 1.5)

plt.plot(x, y1, 'r', linestyle='--')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'b', linestyle='--')
plt.plot([0, 0], [1, 0], ':')
plt.title('Sigmoid Function with Different Intercepts')
plt.show()

# Cost Function
# Unlike linear regression, logistic regression uses a different cost function.
# This is because the sigmoid function has local minima and maxima, which can cause the cost function to be non-convex.
# The cost function for logistic regression is defined as:
# J(theta) = -1/m * sum(y * log(h(x)) + (1 - y) * log(1 - h(x)))
# where m is the number of training examples, y is the actual class label, and h(x) is the predicted probability.

# Using pytorch to implement logistic regression
# Importing necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Defining the dataset
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)
print(y_train.shape)

# Defining the weights and biases
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Defining the model
# hypothesis = 1 / (1 + torch.exp(-x_train.matmul(W) + b))
# usually written as:
hypothesis = torch.sigmoid(x_train.matmul(W) + b)

print(hypothesis)
print(y_train)

# Calculating the cost of a single sample
#-(y_train[0] * torch.log(hypothesis[0]) + 
#   (1 - y_train[0]) * torch.log(1 - hypothesis[0]))

# Calculating the losses of all samples
# losses = -(y_train * torch.log(hypothesis) + 
#          (1 - y_train) * torch.log(1 - hypothesis))

# print(losses)

# Calculating the cost
# cost = losses.mean()
# print(cost)

# Usually done in one line:
cost = F.binary_cross_entropy(hypothesis, y_train)
print(cost)

# Full Implementation
# Defining the weights and biases
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# Defining the optimizer
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Calculating the cost
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    cost = -(y_train * torch.log(hypothesis) + 
             (1 - y_train) * torch.log(1 - hypothesis)).mean()

    # Improving the cost
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # Printing the cost every 100 epochs
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# Testing the model
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)

prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)

# Printing the weights and biases
print(W)
print(b)
