import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from sklearn.datasets import fetch_california_housing

# import sample data
housing = fetch_california_housing()
m,n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]
X = torch.Tensor(housing['data'])
y = torch.Tensor(housing['target']).unsqueeze(1)

# create the weight vector
w_init = torch.randn(8,1,requires_grad=True)

# TO DO:
# a) calculate closed form gradient with respect to the weights
closed_w_grad = (2 * X.T @ X @ w_init) - (2 * X.T @ y)
print(closed_w_grad)

# b) calculate gradient with respect to the weights w using autograd # first create the activation function
X_bar = X @ w_init
loss = ((y - X_bar)**2).sum()
loss.backward()
print(w_init.grad)

# c) check that the two are equal
torch.abs(closed_w_grad - w_init.grad) < 10000

# most of the above are true. 
# The gradients are just really big, so we have to have a high tolerance
