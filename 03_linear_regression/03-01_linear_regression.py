import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# This is a simple linear regression model using PyTorch
def main():
    # Set the random seed for reproducibility
    torch.manual_seed(1)
    # Create a simple dataset
    x_train = torch.FloatTensor([[1], [2], [3]])
    y_train = torch.FloatTensor([[2], [4], [6]])
    # Print the training data
    print(x_train)
    print(x_train.shape)
    print(y_train)
    print(y_train.shape)
    # W is the weight term
    # In this case, we are using a single linear regression model
    # with one input and one output, so we only need one weight and one bias
    # In a more complex model, we might have multiple weights and biases
    # for each input and output
    W = torch.zeros(1, requires_grad=True)
    print(W)
    # b is the bias term
    b = torch.zeros(1, requires_grad=True)
    print(b)
    # The hypothesis is the predicted output
    hypothesis = x_train * W + b
    print(hypothesis)
    # Declaring the cost function
    cost = torch.mean((hypothesis - y_train) ** 2)
    print(cost) 
    # Defining the optimizer
    optimizer = optim.SGD([W, b], lr=0.01)
    
    nb_epochs = 2000
    
    # Training the model
    for epoch in range(nb_epochs):
        # Compute the hypothesis
        hypothesis = x_train * W + b
        # Compute the cost
        cost = torch.mean((hypothesis - y_train) ** 2)
        # Zero the gradients
        # This is important because PyTorch accumulates gradients by default
        # If we don't zero the gradients, the gradients will accumulate
        # and we will get incorrect results
        optimizer.zero_grad()
        # Backpropagation
        cost.backward()
        # Update the weights and bias
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: W={W.item()}, b={b.item()}, cost={cost.item()}')

    # Learning about AutoGrad
    # This is a feature of PyTorch that automatically computes gradients
    # for us, so we don't have to do it manually
    # We can use the .backward() method to compute the gradients
    # and the .grad attribute to access the gradients
    w = torch.tensor(2.0, requires_grad=True)
    y = w ** 2
    z = 2 * y + 5
    z.backward()
    print('gradient of z with respect to w:', w.grad)
    # The gradient of z with respect to w is 8.0

if __name__ == "__main__":
    main()
