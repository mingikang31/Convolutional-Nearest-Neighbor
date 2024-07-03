'''Convolution 1D Nearest Neighbor Layer'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F

class Conv1d_NN(nn.Module):
    def __init__(self, in_channels, out_channels,  K = 3, stride=3, padding=0): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding
        
        # Conv1d Layer
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels, 
                                      out_channels=self.out_channels, 
                                      kernel_size=self.K, 
                                      stride=self.stride, 
                                      padding=self.padding)
                                      

    def forward(self, x):
        
        distance_matrix = self.calculate_distance_matrix(x)
        
        prime_2d = self.prime_vmap_2d(x, distance_matrix, self.K)
        
        # prime_3d = self.prime_vmap_3d(x, distance_matrix, self.K)
        
        # Calculate the convolution 1d         
        return self.conv1d_layer(prime_2d)
    
     
    '''Utility Functions for Nearest Neighbor Tensor'''
    @staticmethod 
    def process_batch(matrix, dist_matrix, num_nearest_neighbors, flatten=True):
        '''Process the batch of matrices''' 
        ind = torch.topk(dist_matrix, num_nearest_neighbors, largest=False).indices
        neigh = matrix[:, ind]
        if flatten: 
            reshape = torch.flatten(neigh, start_dim=1)
            return reshape
        else: 
            return neigh
        
    @staticmethod 
    def calculate_dot_product(matrix): 
        '''Calculate the dot product of the matrix'''
        return torch.bmm(matrix.transpose(2, 1), matrix)

    @staticmethod   
    def calculate_distance_matrix(matrix): 
        '''Calculating the distance matrix of the input matrix'''
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)
        dot_product = Conv1d_NN.calculate_dot_product(matrix)
        dist_matrix = norm_squared + norm_squared.transpose(2, 1) - 2 * dot_product
        return torch.sqrt(dist_matrix)

    @staticmethod 
    def prime_vmap_2d(matrix, dist_matrix, num_nearest_neighbors): 
        # Vectorization / Vmap Implementation for Nearest Neighbor Tensor 2D
        batched_process = torch.vmap(Conv1d_NN.process_batch, in_dims=(0, 0, None), out_dims=0)
        prime = batched_process(matrix, dist_matrix, num_nearest_neighbors, flatten=True)
        return prime 
    
    @staticmethod 
    def prime_vmap_3d(matrix, dist_matrix, num_nearest_neighbors): 
        # Vectorization / Vmap Implementation for Nearest Neighbor Tensor 3D
        batched_process = torch.vmap(Conv1d_NN.process_batch, in_dims=(0, 0, None), out_dims=0)
        prime = batched_process(matrix, dist_matrix, num_nearest_neighbors, flatten=False)
        return prime
    
'''EXAMPLE USAGE'''


layer = Conv1d_NN(12, 32, K =3) # 1 in_channel, 32 out_channel, 40 kernel size, 3 nearest neighbors
ex = torch.rand(32, 12, 40) # 32 samples, 1 channels, 40 tokens

print(ex.shape)

output = layer.forward(ex)
output = Conv1d_NN(12, 32, K=3).forward(ex)

print(output.shape)
print("-"*50)

