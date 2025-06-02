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
from Conv1d_NN import Conv1d_NN


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
    
        super(Conv2d_NN, self).__init__()
        
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
        
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        

        self.pointwise_conv = nn.Conv2d(self.out_channels + 2, self.out_channels, kernel_size=1)
        
        self.coord_cache = {}
        
    def forward(self, x): 
        if self.shuffle_pattern in ["B", "BA"]:
            if self.location_channels: 
                x_coordinates = self.coordinate_channels(x.shape, device=x.device)
                x = torch.cat((x, x_coordinates), dim=1)
                x1 = self.unshuffle_layer(x)
            else: 
                x1 = self.unshuffle_layer(x)
            
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
                x4 = self.shuffle_layer(x4)
                x5 = self.pointwise_conv(x4) ## Added Pointwise Conv to reduce channels added for spatial coordinates
            else:
                x5 = self.shuffle_layer(x4)
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



if __name__ == "__main__":
    x = torch.rand(32, 3, 32, 32)

    conv2d_nn = Conv2d_NN(in_channels=3, out_channels=8, K=3, stride=3, padding=0, shuffle_pattern="BA", shuffle_scale=2, samples="all", magnitude_type="similarity", location_channels=False)
    output = conv2d_nn(x)
    
    print("Input shape:", x.shape) # Should be (32, 3, 32, 32)
    print("Output shape:", output.shape) # Should be (32, 8, 32, 32)
    