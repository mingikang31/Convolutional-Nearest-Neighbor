'''Convolution 1D Nearest Neighbor'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from NNT import NNT

class Conv1d_NN(nn.Module):
    def __init__(self, in_channels, out_channels,  K = 3): 
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels, 
                                      out_channels=self.out_channels, 
                                      kernel_size=K, stride=K)

    def forward(self, x):
        # Create a NNT object
        nnt = NNT(x, self.K)
        
        # Get the convolution matrix
        prime = nnt.prime
         
        # Calculate the convolution 1d         
        return self.conv1d_layer(prime)
    
'''EXAMPLE USAGE'''

layer = Conv1d_NN(1, 32, K =3) # 1 in_channel, 32 out_channel, 40 kernel size, 3 nearest neighbors
ex = torch.rand(32, 1, 40) # 32 samples, 1 channels, 40 tokens

nnt = NNT(ex, 3)
print(nnt.prime.shape)


output = layer.forward(ex)
# output = Conv1d_NN(2, 1, K=3).forward(ex)

print(output.shape)
# print("-"*50)
# print(output)

