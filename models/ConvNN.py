import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Conv2d_NN(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 K, 
                 stride, 
                 padding, 
                 sampling_type, 
                 num_samples, 
                 sample_padding,
                 shuffle_pattern, 
                 shuffle_scale, 
                 magnitude_type, 
                 coordinate_encoding):

        super(Conv2d_NN_sanity, self).__init__()

        assert K == stride, "K must be equal to stride for ConvNN"
        assert padding > 0 or padding == 0, "Cannot have Negative Padding"
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Shuffle pattern must be: Before, After, Before After, Not Applicable"
        assert magnitude_type in ["cosine", "euclidean"], "Similarity Matrix must be either cosine similarity or euclidean distance"
        assert sampling_type in ["all", "random", "spatial"], "Consider all neighbors, random neighbors, or spatial neighbors"
        assert int(num_samples) > 0 or int(num_samples) == -1, "Number of samples to consider must be greater than 0 or -1 for all samples"
        assert (sampling_type == "all" and int(num_samples) == -1) or (sampling_type != "all" and isinstance(num_samples, int)), "Number of samples must be -1 for all samples or integer for random and spatial sampling"

        # Core Parameters
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.K = K
        self.stride = stride 
        self.padding = padding 

        # 3 Sampling Types: all, random, spatial
        self.sampling_type = sampling_type
        self.num_samples = num_samples
        self.sample_padding = sample_padding if sampling_type == "spatial" else 0

        # Pixel Shuffling (optional) 
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale

        # Similarity Metric
        self.magnitude_type = magnitude_type
        self.maximum = True if magnitude_type == "cosine" else False
        self.INF_DISTANCE = 1e10
        self.NEG_INF_DISTANCE = -1e10
        
        # Positional Encoding (optional)
        self.coordinate_encoding = coordinate_encoding 
        self.coordinate_cache = {}

        # Pixel Shuffle Adjustments
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale) 
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)

        self.in_channels_1d = self.in_channels * (self.shuffle_scale ** 2) if self.shuffle_pattern in ["B", "BA"] else self.in_channels
        self.out_channels_1d = self.out_channels * (self.shuffle_scale ** 2) if self.shuffle_pattern in ["A", "BA"] else self.out_channels

        self.in_channels_1d = self.in_channels_1d + 2 if self.coordinate_encoding else self.in_channels_1d

        self.conv1d_layer = nn.Conv1d(
            in_channels = self.in_channels_1d,
            out_channels = self.out_channels_1d,
            kernel_size = self.K, 
            stride = self.stride, 
            padding = 0, 
            bias = False
        )

        self.og_shape = None 
        self.padded_shape = None


    def forward(self, x):  
        # 1. Pixel Shuffle 
        x = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x
        self.og_shape = x.shape

        # 2. Add Padding 
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='replicate')
            self.padded_shape = x.shape

        # 3. Add Coordinate Encoding
        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x

        # 4. Sampling 

        # 5. Similarity Calculation 

        # 6. Aggregation 


        # 7. Pixel Unshuffle

        
        
        

        return 
    

        

        
        