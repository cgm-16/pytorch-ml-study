import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# Implementing logistic regression using nn.Linear and nn.Sigmoid
# nn.Sequential is a container for layers
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

print(model(x_train))

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000

# Training loop
for epoch in range(nb_epochs):
    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        prediction = model(x_train)
        pred = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = pred.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print(f'Epoch {epoch}/{nb_epochs}: Cost: {cost.item():.4f} Accuracy: {accuracy:.4f}')

print(model(x_train))

print(list(model.parameters()))

# Logistic regression can be considered as a neural network with one layer
# However, sigmoid functions are not used as activation functions in neural networks

# Implementing logistic regression using nn.Module

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

model = BinaryClassifier()

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000

# Training loop
for epoch in range(nb_epochs):
    hypothesis = model(x_train)

    cost = F.binary_cross_entropy(hypothesis, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print(f'Epoch {epoch}/{nb_epochs}: Cost: {cost.item():.4f} Accuracy: {accuracy:.4f}')
