'''Convolution 1D Nearest Neighbor'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from NNT import NNT

class Conv1d_NN(nn.Module):
    def __init__(self, in_channels, out_channels,  K = 3, stride=3, padding=0): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding
    
        
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels, 
                                      out_channels=self.out_channels, 
                                      kernel_size=self.K, 
                                      stride=self.stride, 
                                      padding=self.padding)
                                      

    def forward(self, x):
        # Create a NNT object
        nnt = NNT(x, self.K)
        
        # Get the convolution matrix
        prime = nnt.prime_vmap_2d
         
        # Calculate the convolution 1d         
        return self.conv1d_layer(prime)
    
'''EXAMPLE USAGE'''

'''
layer = Conv1d_NN(1, 32, K =3) # 1 in_channel, 32 out_channel, 40 kernel size, 3 nearest neighbors
ex = torch.rand(32, 1, 40) # 32 samples, 1 channels, 40 tokens

nnt = NNT(ex, 3)
print(nnt.prime.shape)


output = layer.forward(ex)
# output = Conv1d_NN(2, 1, K=3).forward(ex)

print(output.shape)
# print("-"*50)
# print(output)
'''



# MNIST-1D Dataset Convolution Neural Network 

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
        x = x.view(-1,1,x.shape[-1])
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        h3 = h3.view(h3.shape[0], -1) # flatten the conv features
        return self.linear(h3) # a linear classifier goes on top
    