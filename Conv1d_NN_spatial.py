'''Convolution 1D Nearest Neighbor Layer'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D
import random 
import time
import faiss
import numpy as np

class Conv1d_NN_spatial(nn.Module): 
   
       def __init__(self, in_channels, out_channels,  K = 3, stride=3, padding=0, shuffle_pattern="N/A", shuffle_scale=2, magnitude_type='distance'): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        
        if samples != "all": 
            self.samples = int(samples)

        else: 
            self.samples = samples
        
        self.magnitude_type = magnitude_type 
        if self.magnitude_type == 'distance': 
            self.maximum = False
        elif self.magnitude_type == 'similarity':
            self.maximum = True            
                    
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
        if self.samples == "all": 
            # Unshuffle Layer
            if self.shuffle_pattern == "B" or self.shuffle_pattern == "BA":
                x1 = self.unshuffle_layer(x)
            else: 
                x1 = x
            
            # Calculate Distance/Similarity Matrix + Prime Vmap 2D
            if self.magnitude_type == 'distance':
                matrix_magnitude = self.calculate_distance_matrix(x1)
            elif self.magnitude_type == 'similarity':
                matrix_magnitude = self.calculate_similarity_matrix(x1)
                
            prime_2d = self.prime_vmap_2d(x1, matrix_magnitude, self.K, self.maximum)
                        
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
            rand_idx = rand_idx = np.random.choice(x1.shape[2], self.samples, replace=False) 
            x1_prime = x1[:, :, rand_idx]
        
            
            if self.magnitude_type == 'distance':
                matrix_magnitude = self.calculate_distance_matrix_N(x1, x1_prime)
            elif self.magnitude_type == 'similarity':
                matrix_magnitude = self.calculate_similarity_matrix_N(x1, x1_prime)
                
            # prime1 = self.prime_brute_N(x1, matrix_magnitude, self.K, rand_idx, self.max)

            if self.magnitude_type == 'distance':
                matrix_magnitude[:, rand_idx, :] = np.inf
            elif self.magnitude_type == 'similarity':
                matrix_magnitude[:, rand_idx, :] = -np.inf

            # Reshape prime to (B, C, N*K)
            prime = self.prime_vmap_2d_N(x1, matrix_magnitude, self.K, rand_idx, self.maximum)
            prime = prime.reshape(x1.shape[0], x1.shape[1], -1)
            
            # print(torch.equal(prime, prime1))
            # print(prime[0, 0:3, :10])
            # print(prime1[0, 0:3, :10])
            
            # tensor([0.8318, 0.8587, 0.3035, 0.8587, 0.8587, 0.8318, 0.3035, 0.8318, 0.3035, 0.7635])
            # tensor([0.8318, 0.7822, 0.3992, 0.8587, 0.7822, 0.4922, 0.3035, 0.4922, 0.3992, 0.7635])

            # Calculate the convolution 1d   
            x2 = self.conv1d_layer(prime)        

            # ReLU Activation
            x3 = self.relu(x2)

            # Shuffle Layer
            if self.shuffle_pattern == "A" or self.shuffle_pattern == "BA":
                x4 = self.shuffle_layer(x3)
            else:
                x4 = x3

            return x4 
            
    ### All Neighbors ###
    '''Distance Matrix Calculations for All Sample'''
    @staticmethod   
    def calculate_distance_matrix(matrix): 
        '''Calculating the distance matrix of the input matrix'''
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)
        dot_product = Conv1d_NN.calculate_dot_product(matrix)
        dist_matrix = norm_squared + norm_squared.transpose(2, 1) - 2 * dot_product
        return dist_matrix # May need to remove torch.sqrt - do not need that computation

    '''Similarity Matrix Calculations for All Sample'''
    @staticmethod 
    def calculate_similarity_matrix(matrix): 
        '''Calculate the similarity matrix of the input matrix'''
        normalized_matrix = F.normalize(matrix, p=2, dim=1) # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
        dot_product = Conv1d_NN.calculate_dot_product(normalized_matrix)
        similarity_matrix = dot_product 
        return similarity_matrix
    
    '''All Sample Methods'''
    @staticmethod 
    def calculate_dot_product(matrix): 
        '''Calculate the dot product of the matrix'''
        return torch.bmm(matrix.transpose(2, 1), matrix)

    @staticmethod 
    def prime_vmap_2d(matrix, magnitude_matrix, num_nearest_neighbors, largest=False): 
        # Vectorization / Vmap Implementation for Nearest Neighbor Tensor 2D
        batched_process = torch.vmap(Conv1d_NN.process_batch, in_dims=(0, 0, None), out_dims=0)
        prime = batched_process(matrix, magnitude_matrix, num_nearest_neighbors, flatten=True, largest=largest)
        return prime 
    
    # @staticmethod 
    # def prime_vmap_3d(matrix, magnitude_matrix , num_nearest_neighbors): 
    #     # Vectorization / Vmap Implementation for Nearest Neighbor Tensor 3D
    #     batched_process = torch.vmap(Conv1d_NN.process_batch, in_dims=(0, 0, None), out_dims=0)
    #     prime = batched_process(matrix, magnitude_matrix, num_nearest_neighbors, flatten=False)
    #     return prime
    
    @staticmethod 
    def process_batch(matrix, magnitude_matrix, num_nearest_neighbors, flatten=True, largest=False):
        '''Process the batch of matrices''' 
        ind = torch.topk(magnitude_matrix, num_nearest_neighbors, largest=largest).indices
        neigh = matrix[:, ind]
        if flatten: 
            reshape = torch.flatten(neigh, start_dim=1)
            return reshape
        else: 
            return neigh
        
    
    ### N Neighbors ###
    '''Distance Matrix Calculations for N Sample'''
    @staticmethod 
    def calculate_distance_matrix_N(m1, m2):
        '''Calculate the distance matrix between two input matrices'''
        # to make it simple -> normalize the rows, normalize across c (channels), 
        norm_squared_1 = torch.sum(m1 ** 2, dim=1, keepdim=True)
        norm_squared_2 = torch.sum(m2 ** 2, dim=1, keepdim=True).transpose(2, 1)
        dot_product = Conv1d_NN.calculate_dot_product_N(m1, m2)

        norm_squared_1 = norm_squared_1.permute(0, 2, 1)  
        norm_squared_2 = norm_squared_2.permute(0, 2, 1)  

        dist_matrix = norm_squared_1 + norm_squared_2 - 2 * dot_product
        return dist_matrix
    
    '''Similarity Matrix Calculations for N Sample'''
    @staticmethod 
    def calculate_similarity_matrix_N(m1, m2): 
        normalized_matrix_1 = F.normalize(m1, p=2, dim=1) # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
        normalized_matrix_2 = F.normalize(m2, p=2, dim=1)
        
        dot_product = Conv1d_NN.calculate_dot_product_N(normalized_matrix_1, normalized_matrix_2)
                
        similarity_matrix = dot_product 
        return similarity_matrix
        
    '''N Sample Methods'''
    @staticmethod 
    def calculate_dot_product_N(m1, m2): 
        '''Calculate the dot product of the matrix'''
        return torch.bmm(m1.transpose(2, 1), m2) 
    
    @staticmethod 
    def prime_brute_N(matrix, magnitude_matrix, num_nearest_neighbors, rand_idx, largest = False):         
        stack_list = [] 

        for i in range(matrix.shape[0]): 
            concat_list = []
            for j in range(matrix.shape[2]): 
                # [1, 28, 30, 14, 15]
                if j in rand_idx: 
                    indices = torch.topk(magnitude_matrix[i, j, :], num_nearest_neighbors, largest=largest).indices
                    indices_list = [rand_idx[i] for i in indices]
                    nearest_neighbors = matrix[i, :, indices_list]
                else: 
                    indices = torch.topk(magnitude_matrix[i, j, :], num_nearest_neighbors - 1, largest=largest).indices
                    indices_list = [j] + [rand_idx[i] for i in indices]
                    nearest_neighbors = matrix[i, :, indices_list]
                    
                concat_list.append(nearest_neighbors)
            
            stacked_neighbors = torch.cat(concat_list, dim=1)
            stack_list.append(stacked_neighbors)
        

        prime = torch.stack(stack_list, dim= 0)
        return prime
    
    @staticmethod 
    def prime_vmap_2d_N(matrix, magnitude_matrix, num_nearest_neighbors, rand_idx, flatten=True, largest=False): 
        '''Vectorization / Vmap Implementation for Nearest Neighbor Tensor 2D'''
        batched_process = torch.vmap(Conv1d_NN.process_batch_N, in_dims=(0, 0, None), out_dims=0)
        prime = batched_process(matrix, magnitude_matrix, num_nearest_neighbors, rand_idx=rand_idx, flatten=flatten, largest=largest)
        return prime

    
    @staticmethod
    def process_batch_N(matrix, magnitude_matrix, num_nearest_neighbors, rand_idx, flatten=True, largest=False):
        '''Process the batch of matrices'''
        
        magnitude_matrix[rand_idx, np.arange(len(rand_idx))]  =  np.inf  


        # Only get num_nearest_neighbors-1 neighbors (we'll add the selves later as the first nearest neighbor)
        _, indxs = torch.topk(magnitude_matrix, num_nearest_neighbors - 1, largest=largest)
        neig = matrix[:, indxs]

        # Add the selves as the first nearest neighbor
        neig = torch.concat([matrix.unsqueeze(2), neig], dim=2)

        if flatten:
            # If flatten is True, flatten the neighbors matrix starting from the second dimension
            neig = torch.flatten(neig, start_dim=1)

        return neig
    
    
    '''Faiss Implementation'''
    @staticmethod 
    def faiss_topk(dist_matrix, num_nearest_neighbors, index=True): 
        '''Faiss Topk Implementation'''
        if index: 
            index = faiss.IndexFlatL2(dist_matrix.shape[2])
            index.add(dist_matrix)
            D, I = index.search(dist_matrix, num_nearest_neighbors)
            return D, I
        else: 
            D, I = faiss.knn(dist_matrix, dist_matrix, num_nearest_neighbors)
            return D, I
        
    
    
'''EXAMPLE USAGE'''

ex = torch.rand(32, 1, 40) # 32 samples, 1 channels, 40 tokens

a = Conv1d_NN(1, 32, K=5, stride=5, padding=0, shuffle_pattern="N/A", shuffle_scale=2, samples="5", magnitude_type="distance").forward(ex)
print(a.shape)
