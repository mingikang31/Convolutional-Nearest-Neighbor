''' Nearest Neighbor Tensor (NNT) Class first prototype'''
import torch
import torch.nn as nn
    
# rename num_closest to num_nearest_neighbors 

class oldNNT: # Nearest Neighbor Tensor
    def __init__(self, matrix, kernel_shape): 
        self.matrix = matrix.to(torch.float32)
        self.kernel_shape = kernel_shape
        
        self.dist_matrix = self.matrix

        self.num_closest = int(kernel_shape[0])
        self.prime = self.prime()
        

    '''Getters for the NNT object'''
    @property 
    def matrix(self): 
        '''Returns the matrix of the NNT object'''
        return self._matrix
    
    @property 
    def kernel_shape(self): 
        '''Returns the kernel of the NNT object'''
        return self._kernel_shape
    
    @property 
    def dist_matrix(self): 
        '''Returns the distance matrix of the NNT object'''
        return self._dist_matrix
    

    @property 
    def num_closest(self): 
        '''Returns the number of closest neighbors to be used in the convolution matrix'''
        return self._num_closest
        
    
    @property 
    def prime(self): 
        '''Returns the convolution matrix of the NNT object'''
        return self._prime
    
    '''Setters for the NNT object'''
    @matrix.setter 
    def matrix(self, value): 
        # Check if the matrix is a torch.Tensor
        if not isinstance(value, torch.Tensor): 
            raise ValueError("Matrix must be a torch.Tensor")
        self._matrix = value
               
    @kernel_shape.setter
    def kernel_shape(self, value): 
        # Check if the kernel shape is a tuple 
        if isinstance(value, tuple): 
            # Check kernel and matrix column size
            if value[1] != self.matrix.shape[2]: 
                raise ValueError("Kernel and Matrix must have the same number of columns")
            
            # Check if the kernel has more rows than the matrix
            if value[0] > self.matrix.shape[1]: 
                raise ValueError("Kernel cannot have more rows than the matrix")
            
            self._kernel_shape = value
        else: 
            raise ValueError("Kernel shape must be a tuple")
        
    @dist_matrix.setter
    def dist_matrix(self, matrix): 
        # Calculate the distance matrix
        self._dist_matrix = torch.zeros(matrix.shape[0], matrix.shape[1], matrix.shape[1])
        
        for i in range(matrix.shape[0]): 
            for j in range(matrix.shape[1]): 
                for k in range(matrix.shape[1]): 
                    self._dist_matrix[i, j, k] = torch.norm(matrix[i, j] - matrix[i, k])
        
    @num_closest.setter
    def num_closest(self, value): 
        self._num_closest = value
    
    def prime(self): 
        stack_list = [] 
        for i in range(self._matrix.shape[0]): 
            
            concat_list = [] 
            for j in range(self._matrix.shape[1]): 
                
                
                # Get the indices of the closest neighbors from the distance matrix
                min_indices = torch.topk(self._dist_matrix[i, :, j], self.num_closest, largest=False).indices
                
                # Get the rows of the matrix that correspond to the closest neighbors
                min_rows = self._matrix[i, min_indices]
                
                # Append the rows to the tensor list for later concatenation 
                concat_list.append(min_rows)
                
            # Concatenate the tensor list to create the convolution matrix
            concat = torch.cat(concat_list, dim=0)
            stack_list.append(concat)
        prime = torch.stack(stack_list, dim=0)
        return prime.squeeze(1)
            
'''EXAMPLE USAGE'''
# ex = torch.rand(3, 10, 2)
# nnt = oldNNT(ex, (3, 2)) # 3 closest neighbors 
# # nnt_error = NNT(ex, (3, 3)) # Error: Kernel and Matrix must have the same number of columns
# print(nnt.prime, "\n\n",  nnt.prime.shape, "\n\n")

