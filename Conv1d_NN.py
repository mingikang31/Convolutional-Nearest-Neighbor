'''Convolution 1D Nearest Neighbor Layer'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D

class Conv1d_NN(nn.Module):
    def __init__(self, in_channels, out_channels,  K = 3, stride=3, padding=0, shuffle_pattern="N/A", shuffle_scale=2, neighbors="all"): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        
        if neighbors != "all": 
            self.neighbors = int(neighbors)
        else: 
            self.neighbors = neighbors
                    
        # Unshuffle Layer 
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)
        
        # Shuffle Layer
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        
        # Conv1d Layer
        if self.shuffle_pattern == "BA" or self.shuffle_pattern == "B": 
            self.in_channels = in_channels  * shuffle_scale
            
        if self.shuffle_pattern == "BA" or self.shuffle_pattern == "A":
            self.out_channels = out_channels * shuffle_scale
        
        
        
        
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels, 
                                      out_channels=self.out_channels, 
                                      kernel_size=self.K, 
                                      stride=self.stride, 
                                      padding=self.padding)
    
        # ReLU Layer
        self.relu = nn.ReLU()
                                      

    def forward(self, x):
        
        # Calculate distance matrix for all neighbors 
        if self.neighbors = "all": 
            # Unshuffle Layer
            if self.shuffle_pattern == "B" or self.shuffle_pattern == "BA":
                x1 = self.unshuffle_layer(x)
            else: 
                x1 = x
            
            # Calculate Distance Matrix + Prime Vmap 2D
            distance_matrix = self.calculate_distance_matrix(x1)
            
            prime_2d = self.prime_vmap_2d(x1, distance_matrix, self.K)
            
            # prime_3d = self.prime_vmap_3d(x, distance_matrix, self.K) # 3D Nearest Neighbor Tensor
            
            # Calculate the convolution 1d   
            x2 = self.conv1d_layer(prime_2d)        

            # ReLU Activation
            x3 = self.relu(x2)

            # Shuffle Layer
            if self.shuffle_pattern == "A" or self.shuffle_pattern == "BA":
                x4 = self.shuffle_layer(x3)
            else:
                x4 = x3

            return x4 
        
        # Calculate distance matrix for N neighbors
        else: 
            # Unshuffle Layer
            if self.shuffle_pattern == "B" or self.shuffle_pattern == "BA":
                x1 = self.unshuffle_layer(x)
            else: 
                x1 = x
            
            # Calculate Distance Matrix + Prime Vmap 2D
            distance_matrix = self.calculate_distance_matrix(x1)
            
            prime_2d = self.prime_vmap_2d(x1, distance_matrix, self.K)
            
            # prime_3d = self.prime_vmap_3d(x, distance_matrix, self.K) # 3D Nearest Neighbor Tensor
            
            # Calculate the convolution 1d   
            x2 = self.conv1d_layer(prime_2d)        

            # ReLU Activation
            x3 = self.relu(x2)

            # Shuffle Layer
            if self.shuffle_pattern == "A" or self.shuffle_pattern == "BA":
                x4 = self.shuffle_layer(x3)
            else:
                x4 = x3

            return x4 
            
    
     
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
        # torch.bmm(matrix.transpose(2,1), matrix_prime) # matrix_prime shape ex. (3, 10) 
        return torch.bmm(matrix.transpose(2, 1), matrix)

    @staticmethod   
    def calculate_distance_matrix(matrix): 
        '''Calculating the distance matrix of the input matrix'''
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)
        dot_product = Conv1d_NN.calculate_dot_product(matrix)
        dist_matrix = norm_squared + norm_squared.transpose(2, 1) - 2 * dot_product
        return dist_matrix # May need to remove torch.sqrt - do not need that computation

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
    
    '''N neighbors calculation functions'''
    @staticmethod 
    def calculate_distance_matrix_N(matrix, )
    
'''EXAMPLE USAGE'''
ex = torch.rand(32, 1, 40) # 32 samples, 1 channels, 40 tokens

# # Before
# B = Conv1d_NN(1, 32, K=3, stride=3, padding=0, shuffle_pattern="B", shuffle_scale=2).forward(ex)
# print("Before", B.shape)

# # After
# A = Conv1d_NN(1, 32, K=3, stride=3, padding=0, shuffle_pattern="A", shuffle_scale=2).forward(ex)
# print("After", A.shape)

# # Before + After 
# BA = Conv1d_NN(1, 32, K=3, stride=3, padding=0, shuffle_pattern="BA", shuffle_scale=2).forward(ex)
# print("Before + After", BA.shape)

# No Shuffle
N = Conv1d_NN(1, 32, K=3, stride=3, padding=0, shuffle_pattern="N/A", shuffle_scale=2).forward(ex)
print("No Shuffle", N.shape)

