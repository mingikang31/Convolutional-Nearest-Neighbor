'''Pixel Shuffle/Unshuffle 1D Layer with Scale Factor'''

import torch
from torch import nn
import torch.nn.functional as F

class PixelShuffle1D(nn.Module): 
    """
    1D Pixel Shuffle Layer for Convolutional Neural Networks.
    
    Attributes: 
        upscale_factor (int): Upscale factor for pixel shuffle. 
        
    Notes:
        Input's channel size must be divisible by the upscale factor. 
    """
    
    def __init__(self, upscale_factor):
        """ 
        Initializes the PixelShuffle1D module.
        
        Parameters:
            upscale_factor (int): Upscale factor for pixel shuffle.
        """
        super(PixelShuffle1D, self).__init__()
        
        self.upscale_factor = upscale_factor

    def forward(self, x): 
        batch_size, channel_len, token_len = x.shape[0], x.shape[1], x.shape[2]
        
        output_channel_len = channel_len / self.upscale_factor 
        if output_channel_len.is_integer() == False: 
            raise ValueError('Input channel length must be divisible by upscale factor')
        output_channel_len = int(output_channel_len)
        
        output_token_len = int(token_len * self.upscale_factor)
        
        x = torch.reshape(x, (batch_size, output_channel_len, output_token_len)).contiguous()
        
        return x 
   
class PixelUnshuffle1D(nn.Module):  
    """
    1D Pixel Unshuffle Layer for Convolutional Neural Networks.
    
    Attributes:
        downscale_factor (int): Downscale factor for pixel unshuffle.
        
    Note:
        Input's token size must be divisible by the downscale factor
    
    """
    
    def __init__(self, downscale_factor):
        """
        Intializes the PixelUnshuffle1D module.
        
        Parameters:
            downscale_factor (int): Downscale factor for pixel unshuffle.
        """
        super(PixelUnshuffle1D, self).__init__()
        
        self.downscale_factor = downscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        channel_len = x.shape[1]
        token_len = x.shape[2]

        output_channel_len = int(channel_len * self.downscale_factor)
        output_token_len = token_len / self.downscale_factor
        
        if output_token_len.is_integer() == False:
            raise ValueError('Input token length must be divisible by downscale factor')
        output_token_len = int(output_token_len)
        
        x = torch.reshape(x, (batch_size, output_channel_len, output_token_len)).contiguous()
        
        return x 



def example_usage():
    '''Example Usage of Pixel Shuffle/Unshuffle 1D Layer'''
    device = 'mps'

    x_test = torch.rand(32, 12, 40).to(device)
    print("Input: ", x_test.shape)

    scale_factor = 4

    pixel_upsample = PixelShuffle1D(scale_factor)
    pixel_downsample = PixelUnshuffle1D(scale_factor)

    x_up = pixel_upsample(x_test)
    print("Upsampled: ", x_up.shape)

    x_up_down = pixel_downsample(x_up)
    print("Downsampled: ", x_up_down.shape)

    if torch.all(torch.eq(x_test, x_up_down)):
        print('Inverse module works.')
