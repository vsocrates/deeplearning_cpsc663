import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class LinearFFN(nn.Module):
    """ Simple feed forward network with one hidden layer."""
    def __init__(self): # initialize the model
        super(LinearFFN, self).__init__() # call for the parent class to initialize
        self.W1 = nn.Parameter(torch.ones(784, 128))
        self.b1 = nn.Parameter(torch.ones(128))

        self.W2 = nn.Parameter(torch.ones(128,10))
        self.b2 = nn.Parameter(torch.ones(10))

    def forward(self, x):
        """
        this is the function that will be executed when we call the logistic regression on data.
        INPUT:
            x, an MNIST image represented as a tensor of shape 784
        OUTPUT:
            predictions, a tensor of shape 10. If using CrossEntropyLoss, 
            your model will be trained to put the largest number in the 
            index it believes corresponds to the correct class.
        """
        # With non-linear activation before
#         out1 = torch.sigmoid(torch.matmul(x,self.W1) + self.b1)
        out1 = torch.matmul(x,self.W1) + self.b1
        predictions = torch.matmul(out1,self.W2) + self.b2

        return predictions
    
import numpy as np

class FFN(nn.Module):
    """ Simple feed forward network with one hidden layer."""
    def __init__(self): # initialize the model
        super(FFN, self).__init__() # call for the parent class to initialize
        
        self.W1 = nn.Parameter(torch.FloatTensor(784, 128).uniform_(-np.sqrt(1/784), np.sqrt(1/784)))
        self.b1 = nn.Parameter(torch.FloatTensor(128).uniform_(-np.sqrt(1/784), np.sqrt(1/784)))

        self.W2 = nn.Parameter(torch.FloatTensor(128, 10).uniform_(-np.sqrt(1/128), np.sqrt(1/128)))
        self.b2 = nn.Parameter(torch.FloatTensor(10).uniform_(-np.sqrt(1/128), np.sqrt(1/128)))

    def forward(self, x):
        """
        this is the function that will be executed when we call the logistic regression on data.
        INPUT:
            x, an MNIST image represented as a tensor of shape 784
        OUTPUT:
            predictions, a tensor of shape 10. If using CrossEntropyLoss, 
            your model will be trained to put the largest number in the 
            index it believes corresponds to the correct class.
        """
        # With non-linear activation before
#         out1 = torch.matmul(x,self.W1) + self.b1
#         out1 = torch.matmul(torch.sigmoid(x),self.W1) + self.b1
#         predictions = torch.matmul(out1,self.W2) + self.b2
        
        out1 = torch.sigmoid(torch.matmul(x,self.W1) + self.b1)
        predictions = torch.matmul(out1,self.W2) + self.b2

        return predictions    
    
    
    