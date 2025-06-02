"""2D Layers for Convolutional Neural Networks."""

"""
# Standalone Layers
(1) Conv2d_NN
(2) Conv2d_NN_Spatial 
(3) Conv2d_NN_Attn
(4) Conv2d_NN_Attn_spatial
(5) Conv2d_NN_Attn_V
(6) Attention2d

# Branching (Conv2d + ConvNN)
(7) Conv2d_ConvNN_Branching
(8) Conv2d_ConvNN_Spatial_Branching
(9) Conv2d_ConvNN_Attn_Branching
(10) Conv2d_ConvNN_Attn_Spatial_Branching
(11) Conv2d_ConvNN_Attn_V_Branching

# Branching (Attention + ConvNN)
(12) Attention_ConvNN_Branching
(13) Attention_ConvNN_Spatial_Branching
(14) Attention_ConvNN_Attn_Branching
(15) Attention_ConvNN_Attn_Spatial_Branching
(16) Attention_ConvNN_Attn_V_Branching

# Branching (Conv2d + Attention)
(17) Attention_Conv2d_Branching
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

from layers1d import * 

"""(1) Conv2d_NN"""
class Conv2d_NN(nn.Module): 
    """
    Convolution 2D Nearest Neighbor Layer
    
    Notes:
        - K must be same as stride. K == stride.
    """
    def __init__(self, 
                in_channels, 
                out_channels, 
                K,
                stride, 
                padding, 
                shuffle_pattern, 
                shuffle_scale, 
                samples, 
                magnitude_type,
                location_channels
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
            location_channels (bool): Whether to add location channels.
        """
        super(Conv2d_NN, self).__init__()
        
        ### Assertions ### 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'" 
        
        # Initialize parameters
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

        # Shuffle2D/Unshuffle2D Layers
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        
        # Adjust Channels for PixelShuffle
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

        # 1D ConvNN Layer
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

        # Flatten Layer
        self.flatten = nn.Flatten(start_dim=2)
        
        # Pointwise Convolution Layer
        self.pointwise_conv = nn.Conv2d(self.out_channels + 2, self.out_channels, kernel_size=1)
        
        # Cache for coordinates
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
                x5 = self.pointwise_conv(x4)
            else:
                x5 = self.shuffle_layer(x4)
        else: 
            if self.location_channels:
                x5 = self.pointwise_conv(x4)
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

"""(2) Conv2d_NN_Spatial"""    
class Conv2d_NN_Spatial(nn.Module): 
    """
        TODO
    """
    def __init__(self, 
                in_channels, 
                out_channels,
                K, 
                stride, 
                padding, 
                shuffle_pattern, 
                shuffle_scale, 
                samples, 
                sample_padding, 
                magnitude_type, 
                location_channels
                ): 
        super(Conv2d_NN_Spatial, self).__init__()
      
    ## Assertions ### 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples > 0 or samples != "all", "Error: samples must be greater than 0 and cannot have 'all' samples'"
      
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = samples
        self.sample_padding = sample_padding
        
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels
        
        # Shuffle2D/Unshuffle2D Layers 
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        
        # Adjust Channels for PixelShuffle
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
                
        # 1D ConvNN Spatial Layer
        self.Conv1d_NN_spatial = Conv1d_NN_Spatial(in_channels=self.in_channels_1d,
                                                    out_channels=self.out_channels_1d,
                                                    K=self.K,
                                                    stride=self.stride,
                                                    padding=self.padding,
                                                    magnitude_type=self.magnitude_type
                                                    )
                                    
        # Flatten Layer
        self.flatten = nn.Flatten(start_dim=2)      
        
        # Pointwise Convolution Layer
        self.pointwise_conv = nn.Conv2d(self.out_channels + 2, self.out_channels, kernel_size=1)
        
        # Cache for coordinates
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
            
        # x sample matrix 
        x_ind = torch.round(torch.linspace(0 + self.sample_padding, x1.shape[2] - self.sample_padding - 1, self.samples)).to(torch.int)
        y_ind = torch.round(torch.linspace(0 + self.sample_padding, x1.shape[3] - self.sample_padding - 1, self.samples)).to(torch.int)
        
        x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
        
        x_idx_flat = x_grid.flatten()
        y_idx_flat = y_grid.flatten()      
                
        width = x1.shape[2]
        # flat indices for indexing -> similar to random sampling for ConvNN
        flat_indices = x_idx_flat * width + y_idx_flat
        
        x_sample = self.flatten(x1[:, :, x_grid, y_grid])
        
        # Input Matrix
        x2 = self.flatten(x1)
        
        x3 = self.Conv1d_NN_spatial(x2, x_sample, flat_indices.to(x.device))
        
        unflatten = nn.Unflatten(dim=2, unflattened_size=x1.shape[2:])
        x4 = unflatten(x3)
        
        if self.shuffle_pattern in ["A", "BA"]:
            if self.location_channels:
                x4 = self.shuffle_layer(x4)
                x5 = self.pointwise_conv(x4)

            else:
                x5 = self.shuffle_layer(x4)
        else: 
            if self.location_channels:
                x5 = self.pointwise_conv(x4)
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

"""(3) Conv2d_NN_Attn"""
class Conv2d_NN_Attn(nn.Module): 
    """
    Convolution 2D Nearest Neighbor Layer for Convolutional Neural Networks.
     
    Notes:
        - K must be same as stride. K == stride.
    """
    
    def __init__(self, 
                in_channels, 
                out_channels, 
                K,
                stride, 
                padding, 
                shuffle_pattern, 
                shuffle_scale, 
                samples, 
                image_size,
                magnitude_type,
                location_channels, 
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
        super(Conv2d_NN_Attn, self).__init__()
        
        ### Assertions ### 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'" 
        assert isinstance(image_size, tuple) and len(image_size) == 2, "Error: image_size must be a tuple of (height, width)"
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = int(samples) if samples != "all" else samples
        self.num_tokens = int((image_size[0] * image_size[1]) / (self.shuffle_scale**2))
        self.image_size = image_size
        
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels
        
        # Shuffle2D/Unshuffle2D Layers
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        
        # Adjust Channels for PixelShuffle
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
        
        # 1D ConvNN Attention Layer
        self.Conv1d_NN_Attn = Conv1d_NN_Attn(in_channels=self.in_channels_1d,
                                    out_channels=self.out_channels_1d,
                                    K=self.K,
                                    stride=self.stride,
                                    padding=self.padding,
                                    samples=self.samples, 
                                    shuffle_pattern="NA",
                                    shuffle_scale=1, 
                                    magnitude_type=self.magnitude_type, 
                                    num_tokens=self.num_tokens
                                    )

        # Flatten Layer
        self.flatten = nn.Flatten(start_dim=2)
        
        # Pointwise Convolution Layer
        self.pointwise_conv = nn.Conv2d(self.out_channels + 2, self.out_channels, kernel_size=1)
        
        # Cache for coordinates
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
        x3 = self.Conv1d_NN_Attn(x2)  

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

"""(4) Conv2d_NN_Attn_Spatial"""
class Conv2d_NN_Attn_Spatial(nn.Module): 
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
                K,
                stride, 
                padding,
                shuffle_pattern,
                shuffle_scale, 
                samples, 
                sample_padding,
                image_size,
                magnitude_type,
                location_channels
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
        
        ### Assertions ### 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'" 
        assert isinstance(image_size, tuple) and len(image_size) == 2, "Error: image_size must be a tuple of (height, width)"
        
        super(Conv2d_NN_Attn_Spatial, self).__init__()
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.num_tokens = int((image_size[0] * image_size[1]) / (self.shuffle_scale**2))
        self.samples = int(samples)
        self.image_size = image_size

        self.magnitude_type = magnitude_type
        self.location_channels = location_channels

        # Shuffle2D/Unshuffle2D Layers
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        
        # Adjust Channels for PixelShuffle
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

        # 1D ConvNN Attention Spatial Layer
        self.Conv1d_NN_Attn_spatial = Conv1d_NN_Attn_Spatial(in_channels=self.in_channels_1d,
                                    out_channels=self.out_channels_1d,
                                    K=self.K,
                                    stride=self.stride,
                                    padding=self.padding,
                                    samples=self.samples**2,
                                    magnitude_type=self.magnitude_type, 
                                    num_tokens=self.num_tokens
                                    )

        # Flatten Layer
        self.flatten = nn.Flatten(start_dim=2)
        
        # Pointwise Convolution Layer
        self.pointwise_conv = nn.Conv2d(self.out_channels + 2, self.out_channels, kernel_size=1)
        
        # Cache for coordinates
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
                
        # x sample_matrix 
        x_ind = torch.round(torch.linspace(0 + self.padding, x1.shape[2] - self.padding - 1, self.samples)).to(torch.int)
        y_ind = torch.round(torch.linspace(0 + self.padding, x1.shape[3] - self.padding - 1, self.samples)).to(torch.int)
        
        x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
        
        x_idx_flat = x_grid.flatten()
        y_idx_flat = y_grid.flatten()
        
        width = x1.shape[2]
        # flat indices for indexing -> similar to random sampling for ConvNN
        flat_indices = x_idx_flat * width + y_idx_flat
        
        x_sample = self.flatten(x1[:, :, x_grid, y_grid])
        
        # Input Matrix
        x2 = self.flatten(x1)

        x3 = self.Conv1d_NN_Attn_spatial(x2, x_sample, flat_indices.to(x.device))  

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

"""(5) Conv2d_NN_Attn_V"""
class Conv2d_NN_Attn_V(nn.Module): 
    """
    Convolution 2D Nearest Neighbor Layer for Convolutional Neural Networks.
        
    Notes:
        - K must be same as stride. K == stride.
    """
    
    def __init__(self, 
                in_channels, 
                out_channels, 
                K,
                stride, 
                padding, 
                shuffle_pattern, 
                shuffle_scale, 
                samples, 
                image_size, 
                magnitude_type,
                location_channels
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
        
        super(Conv2d_NN_Attn_V, self).__init__()
        
        ### Assertions ### 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'" 
        assert isinstance(image_size, tuple) and len(image_size) == 2, "Error: image_size must be a tuple of (height, width)"
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = int(samples) if samples != "all" else samples
        self.num_tokens = int((image_size[0] * image_size[1]) / (self.shuffle_scale**2))
        self.image_size = image_size
        
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels

        # Shuffle2D/Unshuffle2D Layers
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        
        # Adjust Channels for PixelShuffle
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

        # 1D ConvNN Attention V Layer
        self.Conv1d_NN_Attn_V = Conv1d_NN_Attn_V(in_channels=self.in_channels_1d,
                                    out_channels=self.out_channels_1d,
                                    K=self.K,
                                    stride=self.stride,
                                    padding=self.padding,
                                    samples=self.samples, 
                                    shuffle_pattern="NA",
                                    shuffle_scale=1, 
                                    magnitude_type=self.magnitude_type, 
                                    num_tokens=self.num_tokens
                                    )

        # Flatten Layer
        self.flatten = nn.Flatten(start_dim=2)
        
        # Pointwise Convolution Layer
        self.pointwise_conv = nn.Conv2d(self.out_channels + 2, self.out_channels, kernel_size=1)
        
        # Cache for coordinates
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
        x3 = self.Conv1d_NN_Attn_V(x2)  

        unflatten = nn.Unflatten(dim=2, unflattened_size=x1.shape[2:])
        x4 = unflatten(x3)

        if self.shuffle_pattern in ["A", "BA"]:
            if self.location_channels:
                x4 = self.shuffle_layer(x4)
                x5 = self.pointwise_conv(x4) 
            else:
                x5 = self.shuffle_layer(x4)
        else: 
            if self.location_channels:
                x5 = self.pointwise_conv(x4) 
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

"""(6) Attention2d"""
class Attention2d(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 num_heads,
                 shuffle_pattern,
                 shuffle_scale,
                 location_channels,
                 ): 
        super(Attention2d, self).__init__()
        
        ### Assertions ###
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert isinstance(num_heads, int) and num_heads > 0, "Error: num_heads must be a positive integer"
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        
        self.location_channels = location_channels
        
        # Shuffle2D/Unshuffle2D Layers
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        
        # Adjust Channels for PixelShuffle
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
                
        # 1D Attention Layer
        self.Attention1d = Attention1d(in_channels=self.in_channels_1d,
                                        out_channels=self.out_channels_1d,
                                        shuffle_pattern="NA",
                                        shuffle_scale=1,
                                        num_heads=self.num_heads
                                        )
        
        # Flatten Layer
        self.flatten = nn.Flatten(start_dim=2)
        
        # Pointwise Convolution Layer
        self.pointwise_conv = nn.Conv2d(self.out_channels + 2, self.out_channels, kernel_size=1)
        
        # Cache for coordinates
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
        x3 = self.Attention1d(x2)

        unflatten = nn.Unflatten(dim=2, unflattened_size=x1.shape[2:])
        x4 = unflatten(x3)

        if self.shuffle_pattern in ["A", "BA"]:
            if self.location_channels:
                x4 = self.shuffle_layer(x4)
                x5 = self.pointwise_conv(x4)
            else:
                x5 = self.shuffle_layer(x4)
        else: 
            if self.location_channels:
                x5 = self.pointwise_conv(x4)
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

"""# Branching (Conv2d + ConvNN)"""
"""(7) Conv2d_ConvNN_Branching"""
class Conv2d_ConvNN_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio, 
                 kernel_size,
                 K, 
                 shuffle_pattern, 
                 shuffle_scale, 
                 samples, 
                 magnitude_type,
                 location_channels):
        
        super(Conv2d_ConvNN_Branching, self).__init__()
        
        ### Assertions ### 
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'" 
        assert sum(channel_ratio) == 2*out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        # Initialize parameters
        self.in_channels = in_channels 
        self.out_channels = out_channels    
        self.channel_ratio = channel_ratio
        self.kernel_size = kernel_size
        self.K = K
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = samples
        
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels
        
        # Branch1 - Conv2d
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 
                          self.channel_ratio[0], 
                          self.kernel_size, 
                          stride=1, 
                          padding=(self.kernel_size - 1) // 2 if self.kernel_size % 2 == 1 else self.kernel_size // 2
                          ),
                nn.ReLU()
            )
        
        # Branch2 - ConvNN
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN(self.in_channels, 
                          self.channel_ratio[1], 
                          K = self.K, 
                          stride = self.K, 
                          samples = self.samples, 
                          padding=0,
                          shuffle_pattern=self.shuffle_pattern, 
                          shuffle_scale=self.shuffle_scale,
                          magnitude_type=self.magnitude_type,
                          location_channels = self.location_channels), 
                nn.ReLU()
            )
        
        # Pointwise Convolution Layer
        self.pointwise_conv = nn.Conv2d(self.out_channels*2, self.out_channels, 1)

    def forward(self, x):
        
        if self.channel_ratio[0] != 0:
            x1 = self.branch1(x)
        
        if self.channel_ratio[1] != 0:
            x2 = self.branch2(x)
        
        if self.channel_ratio[0] == 0:
            concat = x2
        elif self.channel_ratio[1] == 0:
            concat = x1
        else:
        
            concat = torch.cat([x1, x2], dim=1)
        
        reduce = self.pointwise_conv(concat)
        return reduce

"""(8) Conv2d_ConvNN_Spatial_Branching"""
class Conv2d_ConvNN_Spatial_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio, 
                 kernel_size, 
                 K, 
                 shuffle_pattern, 
                 shuffle_scale, 
                 samples, 
                 magnitude_type,
                 location_channels):
        
        super(Conv2d_ConvNN_Spatial_Branching, self).__init__()
        ### Assertions ###
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'"
        assert sum(channel_ratio) == 2*out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        # Initialize parameters        
        self.in_channels = in_channels 
        self.out_channels = out_channels    
        self.channel_ratio = channel_ratio
        self.kernel_size = kernel_size
        self.K = K        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = samples
        
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels
        
        # Branch1 - Conv2d
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 
                          self.channel_ratio[0], 
                          self.kernel_size, 
                          stride=1, 
                          padding=(self.kernel_size - 1) // 2 if self.kernel_size % 2 == 1 else self.kernel_size // 2
                         ),
                nn.ReLU()
            )
            
        # Branch2 - ConvNN Spatial
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_Spatial(self.in_channels, 
                                  self.channel_ratio[1], 
                                  K=self.K, 
                                  stride=self.K, 
                                  padding=0,
                                  samples=self.samples, 
                                  sample_padding=0,
                                  shuffle_pattern=self.shuffle_pattern, 
                                  shuffle_scale=self.shuffle_scale,
                                  magnitude_type=self.magnitude_type,
                                  location_channels=self.location_channels), 
                nn.ReLU()
            )

        # Pointwise Convolution Layer
        self.pointwise_conv = nn.Conv2d(self.out_channels*2, self.out_channels, 1)

    def forward(self, x):
        
        if self.channel_ratio[0] != 0:
            x1 = self.branch1(x)
        
        if self.channel_ratio[1] != 0:
            x2 = self.branch2(x)
        
        if self.channel_ratio[0] == 0:
            concat = x2
        elif self.channel_ratio[1] == 0:
            concat = x1
        else:
        
            concat = torch.cat([x1, x2], dim=1)
        
        reduce = self.pointwise_conv(concat)
        return reduce

"""(9) Conv2d_ConvNN_Attn_Branching"""
class Conv2d_ConvNN_Attn_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio, 
                 kernel_size, 
                 K,
                 shuffle_pattern, 
                 shuffle_scale, 
                 samples, 
                 image_size,
                 magnitude_type,                  
                 location_channels
                ):
        super(Conv2d_ConvNN_Attn_Branching, self).__init__()

        ### Assertions ###
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'"
        assert isinstance(image_size, tuple) and len(image_size) == 2, "Error: image_size must be a tuple of (height, width)"
        assert sum(channel_ratio) == 2*out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        # Initialize parameters
        self.in_channels = in_channels 
        self.out_channels = out_channels    
        self.channel_ratio = channel_ratio
        self.kernel_size = kernel_size
        self.K = K
        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = samples
        self.image_size = image_size 
        
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels
        
        # Branch1 - Conv2d
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 
                          self.channel_ratio[0], 
                          self.kernel_size, 
                          stride=1, 
                          padding=(self.kernel_size - 1) // 2 if self.kernel_size % 2 == 1 else self.kernel_size // 2
                         ),
                nn.ReLU()
            )
        
        # Branch2 - ConvNN_Attn
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_Attn(self.in_channels, 
                               self.channel_ratio[1], 
                               K=self.K, 
                               stride=self.K, 
                               padding=0,
                               samples=self.samples, 
                               shuffle_pattern=self.shuffle_pattern,
                               shuffle_scale=self.shuffle_scale,    
                               image_size=self.image_size, 
                               magnitude_type=self.magnitude_type,
                               location_channels=self.location_channels), 
                nn.ReLU()
            )
        
        # Pointwise Convolution Layer
        self.pointwise_conv = nn.Conv2d(self.out_channels*2, self.out_channels, 1)

    def forward(self, x):
        if self.channel_ratio[0] != 0:
            x1 = self.branch1(x)
        
        if self.channel_ratio[1] != 0:
            x2 = self.branch2(x)
        
        if self.channel_ratio[0] == 0:
            concat = x2
        elif self.channel_ratio[1] == 0:
            concat = x1
        else:
            concat = torch.cat([x1, x2], dim=1)
        reduce = self.pointwise_conv(concat)
        return reduce
    
"""(10) Conv2d_ConvNN_Attn_Spatial_Branching"""
class Conv2d_ConvNN_Attn_Spatial_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio, 
                 kernel_size, 
                 K, 
                 shuffle_pattern, 
                 shuffle_scale,
                 samples, 
                 image_size, 
                 magnitude_type, 
                 location_channels, 
                 ):
        super(Conv2d_ConvNN_Attn_Spatial_Branching, self).__init__()
        ### Assertions ###
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'"
        assert isinstance(image_size, tuple) and len(image_size) == 2, "Error: image_size must be a tuple of (height, width)"
        assert sum(channel_ratio) == 2*out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        # Initialize parameters        
        self.in_ch = in_channels 
        self.out_ch = out_channels    
        self.channel_ratio = channel_ratio
        self.kernel_size = kernel_size
        self.K = K
        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = samples
        self.image_size = image_size 
        
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels
        
        # Branch1 - Conv2d    
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                nn.Conv2d(self.in_ch, 
                          self.channel_ratio[0], 
                          self.kernel_size, 
                          stride=1, 
                          padding=(self.kernel_size - 1) // 2 if self.kernel_size % 2 == 1 else self.kernel_size // 2
                         ),
                nn.ReLU()
            )
        
        # Branch2 - ConvNN_Attn_Spatial
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_Attn_Spatial(self.in_ch, 
                               self.channel_ratio[1], 
                               K=self.K, 
                               stride=self.K, 
                               padding=0,
                               samples=self.samples, 
                               sample_padding=0,
                               shuffle_pattern=self.shuffle_pattern,
                               shuffle_scale=self.shuffle_scale,    
                               image_size=self.image_size, 
                               magnitude_type=self.magnitude_type,
                               location_channels=self.location_channels), 
                nn.ReLU()
            )
        
        # Pointwise Convolution Layer
        self.pointwise_conv  = nn.Conv2d(self.out_ch*2, self.out_ch, 1)

    def forward(self, x):
        if self.channel_ratio[0] != 0:
            x1 = self.branch1(x)
        if self.channel_ratio[1] != 0:
            x2 = self.branch2(x)
        if self.channel_ratio[0] == 0:
            concat = x2
        elif self.channel_ratio[1] == 0:
            concat = x1
        else:
            concat = torch.cat([x1, x2], dim=1)
        reduce = self.pointwise_conv(concat)
        return reduce

"""(11) Conv2d_ConvNN_Attn_V_Branching"""
class Conv2d_ConvNN_Attn_V_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio, 
                 kernel_size, 
                 K, 
                 shuffle_pattern,
                 shuffle_scale,
                 samples, 
                 image_size,
                 magnitude_type,
                 location_channels,
                ):
        super(Conv2d_ConvNN_Attn_V_Branching, self).__init__()
        ### Assertions ###
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert isinstance(image_size, tuple) and len(image_size) == 2, "Error: image_size must be a tuple of (height, width)"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'"        
        assert sum(channel_ratio) == 2*out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        # Initialize parameters        
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.channel_ratio = channel_ratio
        self.kernel_size = kernel_size
        self.K = K
        
        self.shuffle_pattern = shuffle_pattern 
        self.shuffle_scale = shuffle_scale
        self.samples = samples
        self.image_size = image_size
        
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels
        
        # Branch1 - Conv2d
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 
                          self.channel_ratio[0], 
                          self.kernel_size, 
                          stride=1, 
                          padding=(self.kernel_size - 1) // 2 if self.kernel_size % 2 == 1 else self.kernel_size // 2
                         ),
                nn.ReLU()
            )
            
        # Branch2 - ConvNN_Attn_V
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_Attn_V(self.in_channels, 
                                 self.channel_ratio[1], 
                                 K = self.K, 
                                 stride = self.K, 
                                 padding=0,
                                 samples = self.samples, 
                                 shuffle_pattern=self.shuffle_pattern,
                                 shuffle_scale=self.shuffle_scale,
                                 image_size = self.image_size, 
                                 magnitude_type=self.magnitude_type,
                                 location_channels = self.location_channels
                                ), 
                nn.ReLU()
            )

        # Pointwise Convolution Layer
        self.pointwise_conv  = nn.Conv2d(self.out_channels*2, self.out_channels, 1)

    def forward(self, x):
        if self.channel_ratio[0] != 0:
            x1 = self.branch1(x)
        if self.channel_ratio[1] != 0:
            x2 = self.branch2(x)
        if self.channel_ratio[0] == 0:
            concat = x2
        elif self.channel_ratio[1] == 0:
            concat = x1
        else:
            concat = torch.cat([x1, x2], dim=1)    
        reduce = self.pointwise_conv(concat)
        return reduce

"""(12) Attention_ConvNN_Branching"""
class Attention_ConvNN_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio, 
                 num_heads, 
                 K, 
                 shuffle_pattern,
                 shuffle_scale,
                 samples, 
                 magnitude_type,
                 location_channels):
        super(Attention_ConvNN_Branching, self).__init__()

        ### Assertions ### 
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'"
        assert isinstance(num_heads, int) and num_heads > 0, "Error: num_heads must be a positive integer"
        assert sum(channel_ratio) == 2*out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        
        self.in_channels = in_channels 
        self.out_channels = out_channels    
        self.channel_ratio = channel_ratio
        self.num_heads = num_heads  
        self.K = K
    
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = samples
        
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels
        
        # Branch1 - Attention2d
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                Attention2d(self.in_channels,
                            self.channel_ratio[0], 
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            location_channels=self.location_channels),
                nn.ReLU()
            )
            
        # Branch2 - ConvNN
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN(self.in_channels, 
                          self.channel_ratio[1], 
                          K=self.K, 
                          stride=self.K, 
                          padding=0,
                          samples=self.samples, 
                          shuffle_pattern=self.shuffle_pattern,
                          shuffle_scale=self.shuffle_scale,
                          magnitude_type=self.magnitude_type,
                          location_channels=self.location_channels), 
                nn.ReLU()
            )
        
        # Pointwise Convolution Layer
        self.pointwise_conv = nn.Conv2d(self.out_channels*2, self.out_channels, 1)

    def forward(self, x):
        if self.channel_ratio[0] != 0:
            x1 = self.branch1(x)
        if self.channel_ratio[1] != 0:
            x2 = self.branch2(x)
        if self.channel_ratio[0] == 0:
            concat = x2
        elif self.channel_ratio[1] == 0:
            concat = x1
        else:
            concat = torch.cat([x1, x2], dim=1)
        reduce = self.pointwise_conv(concat)
        return reduce

"""(13) Attention_ConvNN_Spatial_Branching"""
class Attention_ConvNN_Spatial_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio, 
                 num_heads,   
                 K, 
                 shuffle_pattern,  
                 shuffle_scale,  
                 samples,
                 magnitude_type,
                 location_channels):
        super(Attention_ConvNN_Spatial_Branching, self).__init__()
        
        ### Assertions ###
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'"
        assert isinstance(num_heads, int) and num_heads > 0, "Error: num_heads must be a positive integer"
        assert sum(channel_ratio) == 2*out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        # Initializing Parameters
        self.in_channels = in_channels 
        self.out_channels = out_channels    
        self.channel_ratio = channel_ratio
        self.num_heads = num_heads
        self.K = K
        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = samples
        
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels
        
        # Branch1 - Attention2d
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                Attention2d(in_channels=self.in_channels, 
                            out_channels=self.channel_ratio[0], 
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            location_channels=self.location_channels),
                nn.ReLU()
            )
            
        # Branch2 - ConvNN_Spatial
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_Spatial(in_channels=self.in_channels, 
                                  out_channels=channel_ratio[1], 
                                  K=self.K, 
                                  stride=self.K, 
                                  padding=0,
                                  samples=self.samples, 
                                  samples_padding=0,
                                  shuffle_pattern=self.shuffle_pattern,
                                  shuffle_scale=self.shuffle_scale,
                                  magnitude_type=self.magnitude_type,
                                  location_channels=self.location_channels), 
                nn.ReLU()
            )

        # Pointwise Convolution Layer
        self.pointwise_conv  = nn.Conv2d(self.out_channels*2, self.out_channels, 1)

    def forward(self, x):
        if self.channel_ratio[0] != 0:
            x1 = self.branch1(x)
        if self.channel_ratio[1] != 0:
            x2 = self.branch2(x)
        if self.channel_ratio[0] == 0:
            concat = x2
        elif self.channel_ratio[1] == 0:
            concat = x1
        else:
            concat = torch.cat([x1, x2], dim=1)
        reduce = self.pointwise_conv(concat)
        return reduce
        
"""(14) Attention_ConvNN_Attn_Branching"""
class Attention_ConvNN_Attn_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio, 
                 num_heads, 
                 K, 
                 shuffle_pattern, 
                 shuffle_scale, 
                 samples, 
                 image_size,
                 magnitude_type,
                 location_channels):
        
        super(Attention_ConvNN_Attn_Branching, self).__init__()

        ### Assertions ### 
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'"
        assert isinstance(num_heads, int) and num_heads > 0, "Error: num_heads must be a positive integer"
        assert isinstance(image_size, tuple) and len(image_size) == 2, "Error: image_size must be a tuple of (height, width)"
        assert sum(channel_ratio) == 2*out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        # Initialize parameters
        self.in_channels = in_channels 
        self.out_channels = out_channels    
        self.channel_ratio = channel_ratio
        self.num_heads = num_heads
        self.K = K
        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = samples
        self.image_size = image_size
        
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels
        
        # Branch1 - Attention2d
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                Attention2d(in_channels=self.in_channels, 
                            out_channels=self.channel_ratio[0], 
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            location_channels=self.location_channels),
                nn.ReLU()
            )
            
        # Branch2 - ConvNN_Attn
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_Attn(self.in_channels, 
                               self.channel_ratio[1], 
                               K=self.K, 
                               stride=self.K, 
                               padding=0,
                               shuffle_pattern=self.shuffle_pattern,
                               shuffle_scale=self.shuffle_scale,
                               samples=self.samples, 
                               image_size=self.image_size, 
                               magnitude_type=self.magnitude_type,
                               location_channels=self.location_channels), 
                nn.ReLU()
            )

        # Pointwise Convolution Layer
        self.pointwise_conv  = nn.Conv2d(self.out_channels*2, self.out_channels, 1)

    def forward(self, x):
        if self.channel_ratio[0] != 0:
            x1 = self.branch1(x)
        if self.channel_ratio[1] != 0:
            x2 = self.branch2(x)
        if self.channel_ratio[0] == 0:
            concat = x2
        elif self.channel_ratio[1] == 0:
            concat = x1
        else:
            concat = torch.cat([x1, x2], dim=1)
        reduce = self.pointwise_conv(concat)
        return reduce
    
"""(15) Attention_ConvNN_Attn_Spatial_Branching"""
class Attention_ConvNN_Attn_Spatial_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio, 
                 num_heads,
                 K, 
                 shuffle_pattern, 
                 shuffle_scale, 
                 samples, 
                 image_size,
                 magnitude_type,
                 location_channels):
        
        super(Attention_ConvNN_Attn_Spatial_Branching, self).__init__()
        
        ### Assertions ###
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'"
        assert isinstance(num_heads, int) and num_heads > 0, "Error: num_heads must be a positive integer"
        assert isinstance(image_size, tuple) and len(image_size) == 2, "Error: image_size must be a tuple of (height, width)"
        assert sum(channel_ratio) == 2*out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        # Initialize parameters
        self.in_channels = in_channels 
        self.out_channels = out_channels    
        self.channel_ratio = channel_ratio
        self.K = K
        self.samples = samples
        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.num_heads = num_heads
        self.image_size = image_size
        
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels
        
        # Branch1 - Attention2d
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                Attention2d(in_channels=self.in_channels, 
                            out_channels=self.channel_ratio[0], 
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            location_channels=self.location_channels),
                nn.ReLU()
            )
            
        # Branch2 - ConvNN_Attn_Spatial
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_Attn_Spatial(self.in_channels, 
                               self.channel_ratio[1], 
                               K=self.K, 
                               stride=self.K, 
                               padding=0,
                               sample_padding=0, 
                               shuffle_pattern=self.shuffle_pattern,
                               shuffle_scale=self.shuffle_scale,
                               samples=self.samples, 
                               image_size=self.image_size, 
                               magnitude_type=self.magnitude_type,
                               location_channels=self.location_channels), 
                nn.ReLU()
            )

        # Pointwise Convolution Layer
        self.pointwise_conv  = nn.Conv2d(self.out_channels*2, self.out_channels, 1)

    def forward(self, x):
        if self.channel_ratio[0] != 0:
            x1 = self.branch1(x)
        if self.channel_ratio[1] != 0:
            x2 = self.branch2(x)
        if self.channel_ratio[0] == 0:
            concat = x2
        elif self.channel_ratio[1] == 0:
            concat = x1
        else:
            concat = torch.cat([x1, x2], dim=1)
        reduce = self.pointwise_conv(concat)
        return reduce

"""(16) Attention_ConvNN_Attn_V_Branching"""
class Attention_ConvNN_Attn_V_Branching(nn.Module):
    def __init__(self, 
                    in_channels, 
                    out_channels, 
                    channel_ratio, 
                    num_heads, 
                    K, 
                    shuffle_pattern, 
                    shuffle_scale,
                    samples, 
                    image_size, 
                    magnitude_type,  
                    location_channels):
        super(Attention_ConvNN_Attn_V_Branching, self).__init__()

        ### Assertions ###  
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert isinstance(image_size, tuple) and len(image_size) == 2, "Error: image_size must be a tuple of (height, width)"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'"
        assert isinstance(num_heads, int) and num_heads > 0, "Error: num_heads must be a positive integer"
        assert sum(channel_ratio) == 2*out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        # Initialize parameters
        self.in_channels = in_channels 
        self.out_channels = out_channels    
        self.channel_ratio = channel_ratio
        self.num_heads = num_heads
        self.K = K
        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = samples
        self.image_size = image_size
        
        self.location_channels = location_channels
        
        # Branch1 - Attention2d
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                Attention2d(in_channels=self.in_channels, 
                            out_channels=self.channel_ratio[0], 
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            location_channels=self.location_channels),
                nn.ReLU()
            )
            
        # Branch2 - ConvNN_Attn_V
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_Attn_V(self.in_channels, 
                               self.channel_ratio[1], 
                               K=self.K, 
                               stride=self.K, 
                               padding=0,
                               shuffle_pattern=self.shuffle_pattern,
                               shuffle_scale=self.shuffle_scale,
                               samples=self.samples, 
                               image_size=self.image_size, 
                               magnitude_type=magnitude_type,
                               location_channels=self.location_channels), 
                nn.ReLU()
            )

        # Pointwise Convolution Layer
        self.pointwise_conv  = nn.Conv2d(self.out_channels*2, self.out_channels, 1)

    def forward(self, x):
        
        if self.channel_ratio[0] != 0:
            x1 = self.branch1(x)
        
        if self.channel_ratio[1] != 0:
            x2 = self.branch2(x)
        
        if self.channel_ratio[0] == 0:
            concat = x2
        elif self.channel_ratio[1] == 0:
            concat = x1
        else:
        
            concat = torch.cat([x1, x2], dim=1)
        
        reduce = self.pointwise_conv(concat)
        return reduce
    
"""(17) Attention_Conv2d_Branching"""
class Attention_Conv2d_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio, 
                 num_heads, 
                 kernel_size,
                 shuffle_pattern, 
                 shuffle_scale, 
                 location_channels
                 ):
        # Channel_ratio must add up to 2*out_ch

        super(Attention_Conv2d_Branching, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels     
        self.channel_ratio = channel_ratio
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        
        self.location_channels = location_channels

    
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                Attention2d(in_channels=self.in_channels, 
                            out_channels=self.channel_ratio[0], 
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            location_channels=self.location_channels),
                nn.ReLU()
            )
            
        

        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                nn.Conv2d(self.in_channels, 
                          self.channel_ratio[1], 
                          self.kernel_size, 
                          stride=1, 
                          padding=(self.kernel_size - 1) // 2 if self.kernel_size % 2 == 1 else self.kernel_size // 2
                         ),
                nn.ReLU()
            )
        
        # Pointwise Convolution Layer
        self.pointwise_conv  = nn.Conv2d(self.out_channels*2, self.out_channels, 1)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        concat = torch.cat([x1, x2], dim=1)
        reduce = self.pointwise_conv(concat)
        return reduce