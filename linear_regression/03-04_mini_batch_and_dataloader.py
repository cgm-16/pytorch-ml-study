import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = nn.Linear(3, 1)  # input_dim=3, output_dim=1
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs):
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        # Compute the hypothesis
        prediction = model(x_batch)
        # Compute the cost
        cost = F.mse_loss(prediction, y_batch)
        # Zero the gradients
        optimizer.zero_grad()
        # Backpropagation
        cost.backward()
        # Update the weights and bias
        optimizer.step()

    if epoch % 5 == 0:
        print(f'Epoch {epoch}/{nb_epochs}: W={model.weight}, b={model.bias}, cost={cost.item()}')

# Defining a random value for prediction
x_test = torch.FloatTensor([[73, 80, 75]])
# Compute the prediction
pred_y = model(x_test)
print(f'Predicted value for input {x_test} is {pred_y.item()}')

# Learning about custom datasets and dataloaders
# Custom datasets and dataloaders are useful when you have a large dataset that doesn't fit into memory or when you want to apply transformations to the data on-the-fly.
# They allow you to load data in batches, which is essential for training deep learning models efficiently.

# Inheriting from torch.utils.data.Dataset allows you to create a custom dataset class.
# You need to implement the __len__ and __getitem__ methods.
class CustomDataset(Dataset): 
  def __init__(self):
    self.x_data = [[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]]
    self.y_data = [[152], [185], [180], [196], [142]]

  # returns the number of samples in the dataset
  def __len__(self): 
    return len(self.x_data)

  # returns a single sample from the dataset
  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y

# The code below demonstrates how to use the custom dataset and dataloader.
dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = nn.Linear(3, 1)  # input_dim=3, output_dim=1
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs):
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        # Compute the hypothesis
        prediction = model(x_batch)
        # Compute the cost
        cost = F.mse_loss(prediction, y_batch)
        # Zero the gradients
        optimizer.zero_grad()
        # Backpropagation
        cost.backward()
        # Update the weights and bias
        optimizer.step()

    if epoch % 5 == 0:
        print(f'Epoch {epoch}/{nb_epochs}: W={model.weight}, b={model.bias}, cost={cost.item()}')
