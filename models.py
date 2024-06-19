# MNIST-1D Dataset Convolution Neural Network 

import torch
import torch.nn as nn
import torch.nn.functional as F 
from Conv1d_NN import *

class ConvBase(nn.Module):
    def __init__(self, output_size, channels=25, linear_in=125):
        super(ConvBase, self).__init__()
        self.conv1 = nn.Conv1d(1, channels, 5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.linear = nn.Linear(linear_in, output_size) # flattened channels -> 10 (assumes input has dim 50)
        print("Initialized ConvBase model with {} parameters".format(self.count_params()))

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x, verbose=False): # the print statements are for debugging
        # print(x.shape)
        x = x.view(-1,1,x.shape[-1])
        # print(x.shape)
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        h3 = h3.view(h3.shape[0], -1) # flatten the conv features
        return self.linear(h3) # a linear classifier goes on top
    
    
class ConvBase_v2(nn.Module):
    def __init__(self, output_size, channels=25, linear_in=125):
        super(ConvBase_v2, self).__init__()
        self.conv1 = nn.Conv1d(1, channels, 5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.linear = nn.Linear(linear_in, output_size) # flattened channels -> 10 (assumes input has dim 50)
        self.flatten = nn.Flatten()
        print("Initialized ConvBase model with {} parameters".format(self.count_params()))

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x, verbose=False): # the print statements are for debugging
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        h3 = self.flatten(h3) # flatten the conv features
        return self.linear(h3) # a linear classifier goes on top
    
    
# Custom Nearest Neighbor Neural Network
class ConvBase_NN(nn.Module): 
    def __init__(self, output_size, channels=25, linear_in=1000, nearest_neighbor=3):
        super(ConvBase_NN, self).__init__()
        self.conv1 = Conv1d_NN(1, channels, K=nearest_neighbor, stride=nearest_neighbor)
        self.conv2 = Conv1d_NN(channels, channels, K=nearest_neighbor, stride=nearest_neighbor)
        self.conv3 = Conv1d_NN(channels, channels, K=nearest_neighbor, stride=nearest_neighbor)
        self.linear = nn.Linear(linear_in, output_size) # flattened channels -> 10 (assumes input has dim 50)
        self.flatten = nn.Flatten()
        
    def count_params(self): 
        return sum([p.view(-1).shape[0] for p in self.parameters()])
    
    def forward(self, x, verbose=False): # the print statments are for debugging
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        h3 = self.flatten(h3)
        return self.linear(h3)
    
'''*** MUST EDIT'''
# Same num parameter as ConvBase Nearest Neighbor Neural Network 
class ConvBase_NN_v2(nn.Module): 
    def __init__(self, output_size, channels=25, linear_in=1000, nearest_neighbor=3):
        super(ConvBase_NN_v2, self).__init__()
        self.conv1 = Conv1d_NN(1, channels, K=nearest_neighbor, stride=nearest_neighbor)
        self.conv2 = Conv1d_NN(channels, channels, K=nearest_neighbor, stride=nearest_neighbor)
        self.conv3 = Conv1d_NN(channels, channels, K=nearest_neighbor, stride=nearest_neighbor)
        self.linear = nn.Linear(linear_in, output_size) # flattened
        
    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x, verbose=False): # the print statments are for debugging
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        h3 = h3.view(h3.shape[0], -1)
        return self.linear(h3)
        