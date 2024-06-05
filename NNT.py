''' Nearest Neighbor Tensor (NNT) Class'''
import torch
    
class NNT: # Nearest Neighbor Tensor
    def __init__(self, matrix, kernel_shape): 
        self.matrix = matrix.to(torch.float32)
        self.kernel_shape = kernel_shape
        
        self.dist_matrix = self.matrix

        self.num_closest = int(kernel_shape[0])
        self.convolution_matrix = self.convolution_matrix()

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
    def convolution_matrix(self): 
        '''Returns the convolution matrix of the NNT object'''
        return self._convolution_matrix
    
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
            if value[1] != self.matrix.shape[1]: 
                raise ValueError("Kernel and Matrix must have the same number of columns")
            
            # Check if the kernel has more rows than the matrix
            if value[0] > self.matrix.shape[0]: 
                raise ValueError("Kernel cannot have more rows than the matrix")
            
            self._kernel_shape = value
        else: 
            raise ValueError("Kernel shape must be a tuple")
        
    @dist_matrix.setter
    def dist_matrix(self, matrix): 
        # Calculate the distance matrix
        self._dist_matrix = torch.zeros(matrix.shape[0], matrix.shape[0])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[0]):
                self._dist_matrix[i, j] = torch.norm(matrix[i] - matrix[j])
        
    @num_closest.setter
    def num_closest(self, value): 
        self._num_closest = value
    
    def convolution_matrix(self): 
        tensor_list = [] 
        for i in range(self._matrix.shape[0]): 
            # Get the indices of the closest neighbors from the distance matrix
            min_indicies = torch.topk(self._dist_matrix[:, i], self.num_closest, largest=False).indices
            
            # Get the rows of the matrix that correspond to the closest neighbors
            min_rows = self._matrix[min_indicies]
            
            # Append the rows to the tensor list for later concatenation 
            tensor_list.append(min_rows)
            
        # Concatenate the tensor list to create the convolution matrix
        concat = torch.cat(tensor_list, dim=0)
        concat = concat.unsqueeze(0).unsqueeze(1)
        return concat
            

'''Testing and Examples'''


## Example Usage (5, 2) matrix 
ex_matrix = torch.tensor([[2, 1], [5, 2], [1, 0], [4, 6], [3, 8]], dtype=torch.float32)
ex_kernel_shape = (6, 2) # Error
ex_kernel_shape = (3, 2)

## Example usage (4, 3) matrix 
ex_matrix = torch.tensor([[2, 1, 3], [5, 2, 1], [1, 0, 2], [4, 6, 5]])
ex_kernel_shape = (5, 3) # Error 
ex_kernel_shape = (2, 3) 

## Create an instance of the NNT class
example = NNT(ex_matrix, ex_kernel_shape)

print(example.convolution_matrix)
print(example.convolution_matrix.shape)


