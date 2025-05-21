import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Create a simple dataset
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# Define the weights and bias for the model
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Defining the optimizer
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs):
    # Compute the hypothesis
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b
    # Compute the cost
    cost = torch.mean((hypothesis - y_train) ** 2)
    # Zero the gradients
    optimizer.zero_grad()
    # Backpropagation
    cost.backward()
    # Update the weights and bias
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}: w1={w1.item()}, w2={w2.item()}, w3={w3.item()}, b={b.item()}, cost={cost.item()}')

# Defining the training data using 2d tensors
x_train = torch.FloatTensor([[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

#Defining the weights and bias for the model
W = torch.zeros((3, 1), requires_grad=True) # 3 inputs, 1 output
b = torch.zeros(1, requires_grad=True)
# Defining the hypothesis
hypothesis = x_train.matmul(W) + b # matrix multiplication
# Defining the optimizer
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs):
    # Compute the hypothesis
    hypothesis = x_train.matmul(W) + b
    # Compute the cost
    cost = torch.mean((hypothesis - y_train) ** 2)
    # Zero the gradients
    optimizer.zero_grad()
    # Backpropagation
    cost.backward()
    # Update the weights and bias
    optimizer.step()

    print(f'Epoch {epoch}: W={W.squeeze().tolist()}, b={b.item()}, cost={cost.item()}')

# Predicting the output using the trained model
with torch.no_grad():
    new_input = torch.FloatTensor([[75, 85, 72]])
    prediction = new_input.matmul(W) + b
    print(f'Predicted value for input {new_input.squeeze().tolist()}: {prediction.item()}')