import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# Create a simple dataset
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# Defining the model. Simple Linear Regression input_dim=1, output_dim=1.
model = nn.Linear(1,1)

# Print the parameters of the model
print(list(model.parameters()))

# Defining the optimizer. Stochastic Gradient Descent (SGD) with learning rate 0.01.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Using the entire dataset for training. 2000 epochs.
nb_epochs = 2000

for epoch in range(nb_epochs):
    # Compute the hypothesis
    prediction = model(x_train)
    # Compute the cost
    cost = F.mse_loss(prediction, y_train)
    # Zero the gradients
    optimizer.zero_grad()
    # Backpropagation
    cost.backward()
    # Update the weights and bias
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{nb_epochs}: W={model.weight.item()}, b={model.bias.item()}, cost={cost.item()}')

# Defining a random variable for testing the model
x_test = torch.FloatTensor([[4.0]])
# Compute the prediction using the model
pred_y = model(x_test)
# Since y = 2x, the expected output is 8
print(f'Predicted value for input {x_test.item()} is {pred_y.item()}')
print(list(model.parameters()))

# Multiple Linear Regression
# Create a simple dataset with 3 features
x_train = torch.FloatTensor([[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# Defining the model. Simple Linear Regression input_dim=3, output_dim=1.
model = nn.Linear(3, 1)
print(list(model.parameters()))

# Defining the optimizer. Stochastic Gradient Descent (SGD) with learning rate 1e-5.
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

# Training the model for 2000 epochs
nb_epochs = 2000
for epoch in range(nb_epochs):
    # Compute the hypothesis
    prediction = model(x_train)
    # Compute the cost
    cost = F.mse_loss(prediction, y_train)
    # Zero the gradients
    optimizer.zero_grad()
    # Backpropagation
    cost.backward()
    # Update the weights and bias
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{nb_epochs}: W={model.weight}, b={model.bias}, cost={cost.item()}')

# Defining a random variable for testing the model
x_test = torch.FloatTensor([[73, 80, 75]])
# Compute the prediction using the model
pred_y = model(x_test)
print(f'Predicted value for input {x_test} is {pred_y.item()}')

print(list(model.parameters()))

# Defining the simple linear regression model using nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # input_dim=1, output_dim=1

    def forward(self, x):
        return self.linear(x)

# Defining the multiple linear regression model using nn.Module
class MultipleLinearRegressionModel(nn.Module):
    def __init__(self):
        super(MultipleLinearRegressionModel, self).__init__()
        self.linear = nn.Linear(3, 1)  # input_dim=3, output_dim=1

    def forward(self, x):
        return self.linear(x)
    
# Training the simple linear regression model using LinearRegressionModel
model = LinearRegressionModel()

# Defining the dataset
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# Defining the optimizer. Stochastic Gradient Descent (SGD) with learning rate 0.01.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training the model for 2000 epochs
nb_epochs = 2000
for epoch in range(nb_epochs):
    # Compute the hypothesis
    prediction = model(x_train)
    # Compute the cost
    cost = F.mse_loss(prediction, y_train)
    # Zero the gradients
    optimizer.zero_grad()
    # Backpropagation
    cost.backward()
    # Update the weights and bias
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{nb_epochs}: W={model.linear.weight.item()}, b={model.linear.bias.item()}, cost={cost.item()}')

# Training the multiple linear regression model using MultipleLinearRegressionModel
model = MultipleLinearRegressionModel()

# Defining the dataset
x_train = torch.FloatTensor([[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# Defining the optimizer. Stochastic Gradient Descent (SGD) with learning rate 1e-5.
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

# Training the model for 2000 epochs
nb_epochs = 2000
for epoch in range(nb_epochs):
    # Compute the hypothesis
    prediction = model(x_train)
    # Compute the cost
    cost = F.mse_loss(prediction, y_train)
    # Zero the gradients
    optimizer.zero_grad()
    # Backpropagation
    cost.backward()
    # Update the weights and bias
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{nb_epochs}: W={model.linear.weight}, b={model.linear.bias}, cost={cost.item()}')
