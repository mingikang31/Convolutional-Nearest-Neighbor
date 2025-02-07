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
from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D


import faiss 
import numpy as np

class Conv2d_NN(nn.Module): 
    """
    Convolution 2D Nearest Neighbor Layer for Convolutional Neural Networks.
    
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
                magnitude_type="distance"
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



        if (self.shuffle_pattern in ["B", "BA"]):
            self.in_channels_1d = self.in_channels * (self.shuffle_scale **2)
            self.out_channels_1d = self.out_channels * (self.shuffle_scale **2)
        else: 
            self.in_channels_1d = self.in_channels
            self.out_channels_1d = self.out_channels

        self.Conv1d_NN = Conv1d_NN(in_channels=self.in_channels_1d,
                                    out_channels=self.out_channels_1d,
                                    K=self.K,
                                    stride=self.stride,
                                    padding=self.padding,
                                    samples=self.samples, 
                                    shuffle_pattern="N/A",
                                    shuffle_scale=1, 
                                    magnitude_type=self.magnitude_type
                                    )

        self.flatten = nn.Flatten(start_dim=2)
        
    def forward(self, x): 
        if self.shuffle_pattern in ["B", "BA"]:
            x1 = nn.functional.pixel_unshuffle(x, self.shuffle_scale)
        else: 
            x1 = x
            

        x2 = self.flatten(x1)

        x3 = self.Conv1d_NN(x2)  

        unflatten = nn.Unflatten(dim=2, unflattened_size=x1.shape[2:])
        x4 = unflatten(x3)

        if self.shuffle_pattern in ["A", "BA"]:
            x5 = nn.functional.pixel_shuffle(x4, self.shuffle_scale)
        else: 
            x5 = x4

        return x5

'''EXAMPLE USAGE'''
def example_usage():
    """Example Usage of Conv2d_NN Layer"""
    ex = torch.rand(32, 3, 28, 28) 
    print("Input: ", ex.shape)

    conv2d_nn = Conv2d_NN(in_channels=1, out_channels=3, K=3, stride=3, padding=0, shuffle_pattern="BA", shuffle_scale=2, samples=5)
    output = conv2d_nn(ex)
    print("Output: ", output.shape)

        


        