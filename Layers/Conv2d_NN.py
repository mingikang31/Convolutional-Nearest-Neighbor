'''Convolution 2D Nearest Neighbor Layer'''

'''
Features: 
    - K Nearest Neighbors for Consideration. 
    - Calculates Distance/Similarity Matrix for All Samples or N Samples
    - Pixel Shuffle/Unshuffle 2D Layer with Scale Factor
    - Conv1d Layer with Kernel Size, Stride, Padding 
'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from Conv1d_NN import Conv1d_NN, Conv1d_NN_optimized
from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D

import numpy as np

class Conv2d_NN_optimized(nn.Module): 
    """
    Convolution 2D Nearest Neighbor Layer for Convolutional Neural Networks.
     - Location Channels : add coordinates -> unshuffle -> flatten -> ConvNN -> unflatten -> shuffle -> remove coordinate 
    
    Attributes: 
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        K (int): Number of Nearest Neighbors for consideration.
        stride (int): Stride size.
        padding (int): Padding size.
        shuffle_pattern (str): Shuffle pattern.
        shuffle_scale (int): Shuffle scale factor.
        samples (int/str): Number of samples to consider.
        magnitude_type (str): Distance or Similarity.
        
    Notes:
        - K must be same as stride. K == stride.
    """
    
    def __init__(self, 
                in_channels, 
                out_channels, 
                K=3,
                stride=3, 
                padding=0, 
                shuffle_pattern="BA", 
                shuffle_scale=2, 
                samples="all", 
                magnitude_type="similarity",
                location_channels=False
                ): 
        
        """
        Initializes the Conv2d_NN module.
        
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            K (int): Number of Nearest Neighbors for consideration.
            stride (int): Stride size.
            padding (int): Padding size.
            shuffle_pattern (str): Shuffle pattern: "B", "A", "BA".
            shuffle_scale (int): Shuffle scale factor.
            samples (int/str): Number of samples to consider.
            magnitude_type (str): Distance or Similarity.
        """
        # assert K == stride, "K must be same as stride. K == stride"
        # assert shuffle_pattern in ["B", "A", "BA"], "Shuffle pattern must be one of: B, A, BA"
        # assert magnitude_type in ["distance", "similarity"], "Magnitude type must be one of: distance, similarity"
        # assert isinstance(samples, (int, str)), "Samples must be int or str"
        # assert isinstance(location_channels, bool), "Location channels must be boolean"
        # assert isinstance(in_channels, int), "Input channels must be int"
        # assert isinstance(out_channels, int), "Output channels must be int"
        # assert isinstance(K, int), "K must be int"
        # assert isinstance(stride, int), "Stride must be int"
        # assert isinstance(padding, int), "Padding must be int"
        # assert isinstance(shuffle_scale, int), "Shuffle scale must be int"
        # assert isinstance(shuffle_pattern, str), "Shuffle pattern must be str"
        # assert isinstance(magnitude_type, str), "Magnitude type must be str"
        
        
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = int(samples) if samples != "all" else samples
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels

        if (self.shuffle_pattern in ["B", "BA"]):
            if self.location_channels: 
                self.in_channels_1d = (self.in_channels + 2) * (self.shuffle_scale**2)
                self.out_channels_1d = (self.out_channels + 2) * (self.shuffle_scale **2)
            else:
                self.in_channels_1d = self.in_channels * (self.shuffle_scale**2)
                self.out_channels_1d = self.out_channels * (self.shuffle_scale **2)
        else: 
            if self.location_channels: 
                self.in_channels_1d = self.in_channels + 2
                self.out_channels_1d = self.out_channels + 2
            else:
                self.in_channels_1d = self.in_channels
                self.out_channels_1d = self.out_channels


        self.Conv1d_NN = Conv1d_NN_optimized(in_channels=self.in_channels_1d,
                                    out_channels=self.out_channels_1d,
                                    K=self.K,
                                    stride=self.stride,
                                    padding=self.padding,
                                    samples=self.samples, 
                                    shuffle_pattern="NA",
                                    shuffle_scale=1, 
                                    magnitude_type=self.magnitude_type
                                    )

        self.flatten = nn.Flatten(start_dim=2)
        

        self.pointwise_conv = nn.Conv2d(self.out_channels + 2, self.out_channels, kernel_size=1)
        
        self.coord_cache = {}
        
    def forward(self, x): 
        if self.shuffle_pattern in ["B", "BA"]:
            if self.location_channels: 
                x_coordinates = self.coordinate_channels(x.shape, device=x.device)
                x = torch.cat((x, x_coordinates), dim=1)
                x1 = nn.functional.pixel_unshuffle(x, self.shuffle_scale)
            else: 
                x1 = nn.functional.pixel_unshuffle(x, self.shuffle_scale)
            
        else: 
            if self.location_channels:
                x_coordinates = self.coordinate_channels(x.shape, device=x.device)
                x1 = torch.cat((x, x_coordinates), dim=1)
            else: 
                x1 = x
                
            
        x2 = self.flatten(x1)

        x3 = self.Conv1d_NN(x2)  

        unflatten = nn.Unflatten(dim=2, unflattened_size=x1.shape[2:])
        x4 = unflatten(x3)

        if self.shuffle_pattern in ["A", "BA"]:
            if self.location_channels:
                x4 = nn.functional.pixel_shuffle(x4, self.shuffle_scale)
                x5 = self.pointwise_conv(x4) ## Added Pointwise Conv to reduce channels added for spatial coordinates
            else:
                x5 = nn.functional.pixel_shuffle(x4, self.shuffle_scale)
        else: 
            if self.location_channels:
                x5 = self.pointwise_conv(x4) ## Added Pointwise Conv to reduce channels added for spatial coordinates
            else: 
                x5 = x4

        return x5
    
    def coordinate_channels(self, tensor_shape, device):
        cache_key = f"{tensor_shape[2]}_{tensor_shape[3]}_{device}"
        if cache_key in self.coord_cache:
            return self.coord_cache[cache_key]
        
        
        x_ind = torch.arange(0, tensor_shape[2])
        y_ind = torch.arange(0, tensor_shape[3])
        
        x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
        
        x_grid = x_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        y_grid = y_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        
        xy_grid = torch.cat((x_grid, y_grid), dim=1)
        xy_grid_normalized = F.normalize(xy_grid, p=2, dim=1)
        self.coord_cache[cache_key] = xy_grid_normalized.to(device)
        
        return xy_grid_normalized.to(device)    


class Conv2d_NN(nn.Module): 
    """
    Convolution 2D Nearest Neighbor Layer for Convolutional Neural Networks.
     - Location Channels : add coordinates -> unshuffle -> flatten -> ConvNN -> unflatten -> shuffle -> remove coordinate 
    
    Attributes: 
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        K (int): Number of Nearest Neighbors for consideration.
        stride (int): Stride size.
        padding (int): Padding size.
        shuffle_pattern (str): Shuffle pattern.
        shuffle_scale (int): Shuffle scale factor.
        samples (int/str): Number of samples to consider.
        magnitude_type (str): Distance or Similarity.
        
    Notes:
        - K must be same as stride. K == stride.
    """
    
    def __init__(self, 
                in_channels, 
                out_channels, 
                K=3,
                stride=3, 
                padding=0, 
                shuffle_pattern="BA", 
                shuffle_scale=2, 
                samples="all", 
                magnitude_type="similarity",
                location_channels=False
                ): 
        
        """
        Initializes the Conv2d_NN module.
        
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            K (int): Number of Nearest Neighbors for consideration.
            stride (int): Stride size.
            padding (int): Padding size.
            shuffle_pattern (str): Shuffle pattern: "B", "A", "BA".
            shuffle_scale (int): Shuffle scale factor.
            samples (int/str): Number of samples to consider.
            magnitude_type (str): Distance or Similarity.
        """
        # assert K == stride, "K must be same as stride. K == stride"
        # assert shuffle_pattern in ["B", "A", "BA"], "Shuffle pattern must be one of: B, A, BA"
        # assert magnitude_type in ["distance", "similarity"], "Magnitude type must be one of: distance, similarity"
        # assert isinstance(samples, (int, str)), "Samples must be int or str"
        # assert isinstance(location_channels, bool), "Location channels must be boolean"
        # assert isinstance(in_channels, int), "Input channels must be int"
        # assert isinstance(out_channels, int), "Output channels must be int"
        # assert isinstance(K, int), "K must be int"
        # assert isinstance(stride, int), "Stride must be int"
        # assert isinstance(padding, int), "Padding must be int"
        # assert isinstance(shuffle_scale, int), "Shuffle scale must be int"
        # assert isinstance(shuffle_pattern, str), "Shuffle pattern must be str"
        # assert isinstance(magnitude_type, str), "Magnitude type must be str"
        
        
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = int(samples) if samples != "all" else samples
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels

        if (self.shuffle_pattern in ["B", "BA"]):
            if self.location_channels: 
                self.in_channels_1d = (self.in_channels + 2) * (self.shuffle_scale**2)
                self.out_channels_1d = (self.out_channels + 2) * (self.shuffle_scale **2)
            else:
                self.in_channels_1d = self.in_channels * (self.shuffle_scale**2)
                self.out_channels_1d = self.out_channels * (self.shuffle_scale **2)
        else: 
            if self.location_channels: 
                self.in_channels_1d = self.in_channels + 2
                self.out_channels_1d = self.out_channels + 2
            else:
                self.in_channels_1d = self.in_channels
                self.out_channels_1d = self.out_channels


        self.Conv1d_NN = Conv1d_NN(in_channels=self.in_channels_1d,
                                    out_channels=self.out_channels_1d,
                                    K=self.K,
                                    stride=self.stride,
                                    padding=self.padding,
                                    samples=self.samples, 
                                    shuffle_pattern="NA",
                                    shuffle_scale=1, 
                                    magnitude_type=self.magnitude_type
                                    )

        self.flatten = nn.Flatten(start_dim=2)
        
        self.pointwise_conv = nn.Conv2d(self.out_channels + 2, self.out_channels, kernel_size=1)
        
        
    def forward(self, x): 
        if self.shuffle_pattern in ["B", "BA"]:
            if self.location_channels: 
                x_coordinates = self.coordinate_channels(x.shape, device=x.device)
                x = torch.cat((x, x_coordinates), dim=1)
                x1 = nn.functional.pixel_unshuffle(x, self.shuffle_scale)
            else: 
                x1 = nn.functional.pixel_unshuffle(x, self.shuffle_scale)
            
        else: 
            if self.location_channels:
                x_coordinates = self.coordinate_channels(x.shape, device=x.device)
                x1 = torch.cat((x, x_coordinates), dim=1)
            else: 
                x1 = x
                
            
        x2 = self.flatten(x1)

        x3 = self.Conv1d_NN(x2)  

        unflatten = nn.Unflatten(dim=2, unflattened_size=x1.shape[2:])
        x4 = unflatten(x3)

        if self.shuffle_pattern in ["A", "BA"]:
            if self.location_channels:
                x4 = nn.functional.pixel_shuffle(x4, self.shuffle_scale)
                x5 = self.pointwise_conv(x4) ## Added Pointwise Conv to reduce channels added for spatial coordinates
            else:
                x5 = nn.functional.pixel_shuffle(x4, self.shuffle_scale)
        else: 
            if self.location_channels:
                x5 = self.pointwise_conv(x4) ## Added Pointwise Conv to reduce channels added for spatial coordinates
            else: 
                x5 = x4

        return x5
    
    def coordinate_channels(self, tensor_shape, device):
        x_ind = torch.arange(0, tensor_shape[2])
        y_ind = torch.arange(0, tensor_shape[3])
        
        x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
        
        x_grid = x_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        y_grid = y_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        
        xy_grid = torch.cat((x_grid, y_grid), dim=1)
        xy_grid_normalized = F.normalize(xy_grid, p=2, dim=1)
        return xy_grid_normalized.to(device)
    
    
    
    
class Conv2d_NN_prev(nn.Module): 
    """
    Convolution 2D Nearest Neighbor Layer for Convolutional Neural Networks.
    - Location Channels : unshuffle -> add coordinates -> flatten -> ConvNN -> unflatten -> remove coordinate -> shuffle
    
    Attributes: 
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        K (int): Number of Nearest Neighbors for consideration.
        stride (int): Stride size.
        padding (int): Padding size.
        shuffle_pattern (str): Shuffle pattern.
        shuffle_scale (int): Shuffle scale factor.
        samples (int/str): Number of samples to consider.
        magnitude_type (str): Distance or Similarity.
        
    Notes:
        - K must be same as stride. K == stride.
    """
    
    def __init__(self, 
                in_channels, 
                out_channels, 
                K=3, 
                stride=3, 
                padding=0, 
                shuffle_pattern="BA", 
                shuffle_scale=2, 
                samples="all", 
                magnitude_type="similarity",
                location_channels=False
                ): 
        
        """
        Initializes the Conv2d_NN module.
        
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            K (int): Number of Nearest Neighbors for consideration.
            stride (int): Stride size.
            padding (int): Padding size.
            shuffle_pattern (str): Shuffle pattern: "B", "A", "BA".
            shuffle_scale (int): Shuffle scale factor.
            samples (int/str): Number of samples to consider.
            magnitude_type (str): Distance or Similarity.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = int(samples) if samples != "all" else samples
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels

        if (self.shuffle_pattern in ["B", "BA"]):
            self.in_channels_1d = self.in_channels * (self.shuffle_scale**2)
            self.out_channels_1d = self.out_channels * (self.shuffle_scale **2)
        else: 
            self.in_channels_1d = self.in_channels
            self.out_channels_1d = self.out_channels
            
        if self.location_channels:
            self.in_channels_1d += 2
            self.out_channels_1d += 2


        self.Conv1d_NN = Conv1d_NN(in_channels=self.in_channels_1d,
                                    out_channels=self.out_channels_1d,
                                    K=self.K,
                                    stride=self.stride,
                                    padding=self.padding,
                                    samples=self.samples, 
                                    shuffle_pattern="NA",
                                    shuffle_scale=1, 
                                    magnitude_type=self.magnitude_type
                                    )

        self.flatten = nn.Flatten(start_dim=2)
        
        self.pointwise_conv = nn.Conv2d(self.out_channels + 2, self.out_channels, kernel_size=1)

        
    def forward(self, x): 
        if self.shuffle_pattern in ["B", "BA"]:
            if self.location_channels: 
                x1 = nn.functional.pixel_unshuffle(x, self.shuffle_scale)
                x1_coordinates = self.coordinate_channels(x1.shape, device=x.device)
                x1 = torch.cat((x1, x1_coordinates), dim=1)
            else: 
                x1 = nn.functional.pixel_unshuffle(x, self.shuffle_scale)
            
        else: 
            if self.location_channels:
                x1_coordinates = self.coordinate_channels(x.shape, device=x.device)
                x1 = torch.cat((x, x1_coordinates), dim=1)
            else: 
                x1 = x
                
            
        x2 = self.flatten(x1)

        x3 = self.Conv1d_NN(x2)  

        unflatten = nn.Unflatten(dim=2, unflattened_size=x1.shape[2:])
        x4 = unflatten(x3)

        if self.shuffle_pattern in ["A", "BA"]:
            if self.location_channels:
                x4 = self.pointwise_conv(x4)
                x5 = nn.functional.pixel_shuffle(x4, self.shuffle_scale)
            else:
                x5 = nn.functional.pixel_shuffle(x4, self.shuffle_scale)
        else: 
            if self.location_channels:
                x5 = self.pointwise_conv(x4)
            else: 
                x5 = x4

        return x5
    
    def coordinate_channels(self, tensor_shape, device):
        x_ind = torch.arange(0, tensor_shape[2])
        y_ind = torch.arange(0, tensor_shape[3])
        
        x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
        
        x_grid = x_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        y_grid = y_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        
        xy_grid = torch.cat((x_grid, y_grid), dim=1)
        xy_grid_normalized = F.normalize(xy_grid, p=2, dim=1)
        return xy_grid_normalized.to(device)

'''EXAMPLE USAGE'''
def example_usage():
    """Example Usage of Conv2d_NN Layer"""
    ex = torch.rand(32, 3, 28, 28) 
    print("Input: ", ex.shape, '\n')

    conv2d_nn = Conv2d_NN(in_channels=3, out_channels=3, K=3, stride=3, padding=0, shuffle_pattern="B", shuffle_scale=2, samples=5,magnitude_type="similarity", location_channels=False)
    output = conv2d_nn(ex)
    print("Output: ", output.shape, '\n')
    
    a = conv2d_nn.coordinate_channels(ex.shape, device=ex.device)
    print("location_channels: ", a.shape, '\n')
    
if __name__ == "__main__":
    example_usage()
    