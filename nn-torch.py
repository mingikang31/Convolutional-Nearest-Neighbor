''' Nearest neighbor search with Pytorch Tensors '''
import torch


    
class NNT: # Nearest Neighbor Tensor
    def __init__(self, matrix, kernel): 
        self._matrix = matrix 
        self._kernel = kernel

        # Calculate the distance matrix
        self._dist_matrix = torch.zeros(matrix.shape[0], matrix.shape[0])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[0]):
                self._dist_matrix[i, j] = torch.norm(matrix[i] - matrix[j])

        self.num_closest = int(kernel.shape[0])
        self._convolution_matrix = self.calc_convolution_matrix()

    @property 
    def matrix(self): 
        '''Returns the matrix of the NNT object'''
        return self._matrix
    
    @property 
    def kernel(self): 
        '''Returns the kernel of the NNT object'''
        return self._kernel
    
    @property 
    def dist_matrix(self): 
        '''Returns the distance matrix of the NNT object'''
        return self._dist_matrix
    

    @property 
    def num_closest(self): 
        '''Returns the number of closest neighbors to be used in the convolution matrix'''
        return self._num_closest
        
    
    @property 
    def convolution_matrix(self): 
        '''Returns the convolution matrix of the NNT object'''
        return self._convolution_matrix
    
    @matrix.setter 
    def matrix(self, value): 
        # Check if the matrix is a torch.Tensor
        if not isinstance(value, torch.Tensor): 
            raise ValueError("Matrix must be a torch.Tensor")
        self._matrix = value
        
    @kernel.setter
    def kernel(self, value): 
        # Check if the kernel is a torch.Tensor
        if not isinstance(value, torch.Tensor): 
            raise ValueError("Kernel must be a torch.Tensor")
        
        # No Errors 
        self._kernel = value
        
    @num_closest.setter
    def num_closest(self, value): 
        # Check if the number of closest neighbors is less than the number of rows in the matrix
        if value > self._matrix.shape[0]:
            raise ValueError("Number of closest neighbors cannot exceed the number of rows in the matrix")
        self._num_closest = value
        
    def calc_convolution_matrix(self): 
        tensor_list = [] 
        for i in range(self._matrix.shape[0]): 
            # Get the indices of the closest neighbors from the distance matrix
            min_indicies = torch.topk(self._dist_matrix[:, i], self.num_closest, largest=False).indices
            
            # Get the rows of the matrix that correspond to the closest neighbors
            min_rows = self._matrix[min_indicies]
            
            # Append the rows to the tensor list for later concatenation 
            tensor_list.append(min_rows)
            
        # Concatenate the tensor list to create the convolution matrix
        return torch.cat(tensor_list, dim=0)
            
            
        
## Example Usage
ex_matrix = torch.tensor([[2, 1], [5, 2], [1, 0], [4, 6], [3, 8]], dtype=torch.float32)
ex_kernel = torch.tensor([[1, 0], [0, 1], [2, 5]], dtype=torch.float32)

# ex_kernel = torch.tensor([[1, 0], [0, 1], [2, 5], [2, 1], [3, 2], [2, 3]], dtype=torch.float32) # For testing

# example = NNT(ex_matrix, ex_kernel)

# print(example.convolution_matrix)

# import torch.nn as nn
# nn.Conv1d(1, 1, 3, 1, 1)