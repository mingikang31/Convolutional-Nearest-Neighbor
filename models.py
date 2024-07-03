# MNIST-1D Dataset Convolution Neural Network 

import torch
import torch.nn as nn
import torch.nn.functional as F 
from Conv1d_NN import Conv1d_NN
from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D


'''Regular Classification Models'''
class ConvBase(nn.Module):
    def __init__(self, output_size, channels=25, linear_in=125):
        super(ConvBase, self).__init__()
        self.conv1 = nn.Conv1d(1, channels, 5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.linear = nn.Linear(linear_in, output_size) # flattened channels -> 10 (assumes input has dim 50)
        print("Initialized ConvBase model with {} parameters".format(self.count_params()))

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x, verbose=False): # the print statements are for debugging
        # print(x.shape)
        x = x.view(-1,1,x.shape[-1])
        # print(x.shape)
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        h3 = h3.view(h3.shape[0], -1) # flatten the conv features
        return self.linear(h3) # a linear classifier goes on top
      
class ConvBase_v2(nn.Module):
    def __init__(self, output_size, channels=25, linear_in=125):
        super(ConvBase_v2, self).__init__()
        self.conv1 = nn.Conv1d(1, channels, 5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.linear = nn.Linear(linear_in, output_size) # flattened channels -> 10 (assumes input has dim 50)
        self.flatten = nn.Flatten()
        print("Initialized ConvBase model with {} parameters".format(self.count_params()))

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x, verbose=False): # the print statements are for debugging
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        h3 = self.flatten(h3) # flatten the conv features
        return self.linear(h3) # a linear classifier goes on top
    
# Custom Nearest Neighbor Neural Network
class ConvBase_NN(nn.Module): 
    def __init__(self, output_size, channels=25, linear_in=1000, nearest_neighbor=3):
        super(ConvBase_NN, self).__init__()
        self.conv1 = Conv1d_NN(1, channels, K=nearest_neighbor, stride=nearest_neighbor)
        self.conv2 = Conv1d_NN(channels, channels, K=nearest_neighbor, stride=nearest_neighbor)
        self.conv3 = Conv1d_NN(channels, channels, K=nearest_neighbor, stride=nearest_neighbor)
        self.linear = nn.Linear(linear_in, output_size) # flattened channels -> 10 (assumes input has dim 50)
        self.flatten = nn.Flatten()
        
    def count_params(self): 
        return sum([p.view(-1).shape[0] for p in self.parameters()])
    
    def forward(self, x, verbose=False): # the print statments are for debugging
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        h3 = self.flatten(h3)
        return self.linear(h3)
    
# Branching Network with Local (Conv1d) + Global Layer (Conv1d_NN)
class BranchingNetwork(nn.Module):
    def __init__(self, in_ch, out_ch1, out_ch2, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch1, kernel_size),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            Conv1d_NN(in_ch, out_ch2, K = kernel_size, stride = kernel_size),
            nn.ReLU()
        )
        self.reduce_channels = nn.Conv1d(out_ch1 + out_ch2, (out_ch1 + out_ch2) // 2, 1)


    def forward(self, x):
        
        x1 = self.branch1(x)
        
        x2 = self.branch2(x)
        
        ## Calculate expected Output size of x2 
        expected_x1_size = x2.size(2) 
        # print(expected_x1_size)
        
        ## Calculate padding for x1 to match x2's size   
        total_padding = expected_x1_size - x1.size(2)
        # print(total_padding)
        
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        
        ## Apply dynamic padding to x1
        x1 = F.pad(x1, (left_padding, right_padding), 'constant', 0)
        
        ## Concatenate the outputs along the channel dimension
        concat = torch.cat([x1, x2], dim=1)
        # print(concat.shape)
        
        ## Reduce the number of channels
        reduce = self.reduce_channels(concat)
        # print(reduce.shape)
        return reduce
    
# Branching Network with Local (Conv1d) + Global Layer (Conv1d_NN) + Pixel Shuffle 
class BranchingNetwork_pixelshuffle(nn.Module):
    def __init__(self, in_ch, out_ch1, out_ch2, kernel_size, upscale_factor):        
        super().__init__()
        self.kernel_size = kernel_size
        self.upscale_factor = upscale_factor
        
        
        
        
        
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch1, kernel_size),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            Conv1d_NN(in_ch, out_ch2, K = kernel_size, stride = kernel_size),
            nn.ReLU()
        )
        self.reduce_channels = nn.Conv1d(out_ch1 + out_ch2, (out_ch1 + out_ch2) // 2, 1)


    def forward(self, x):
        
        x1 = self.branch1(x)
        
        x2 = self.branch2(x)
        
        ## Calculate expected Output size of x2 
        expected_x1_size = x2.size(2) 
        # print(expected_x1_size)
        
        ## Calculate padding for x1 to match x2's size   
        total_padding = expected_x1_size - x1.size(2)
        # print(total_padding)
        
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        
        ## Apply dynamic padding to x1
        x1 = F.pad(x1, (left_padding, right_padding), 'constant', 0)
        
        ## Concatenate the outputs along the channel dimension
        concat = torch.cat([x1, x2], dim=1)
        # print(concat.shape)
        
        ## Reduce the number of channels
        reduce = self.reduce_channels(concat)
        # print(reduce.shape)
        return reduce
    
# U-Net Architecture 
class UNet_pixelshuffle(nn.Module): 
    def __init__(self, in_ch, out_ch, kernel_size, upscale_factor): 
        super(UNet_pixelshuffle, self).__init__()
        self.kernel_size = kernel_size
        self.upscale_factor = upscale_factor
        
        
        
        
        
        
        
        self.down1 = BranchingNetwork_pixelshuffle(in_ch, 16, 16, kernel_size)
        self.down2 = BranchingNetwork_pixelshuffle(8, 32, 32, kernel_size)
        self.down3 = BranchingNetwork_pixelshuffle(16, 64, 64, kernel_size)
        self.down4 = BranchingNetwork_pixelshuffle(32, 128, 128, kernel_size)
        
        self.up1 = BranchingNetwork_pixelshuffle(192, 64, 64, kernel_size)
        self.up2 = BranchingNetwork_pixelshuffle(96, 32, 32, kernel_size)
        self.up3 = BranchingNetwork_pixelshuffle(48, 16, 16, kernel_size)
        self.up4 = BranchingNetwork_pixelshuffle(24, 8, 8, kernel_size)
        
        self.conv = nn.Conv1d(8, out_ch, 1)
    

'''Denoising Models'''
class ConvDenoiser(nn.Module):
    def __init__(self, channels=32):
        super(ConvDenoiser, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=5 , stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=channels, out_channels=1, kernel_size=5, stride=1, padding=2)
        print("Initialized ConvDenoiser model with {} parameters".format(self.count_params()))

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x, verbose=False): # the print statements are for debugging
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        return h3
    
    
class Conv1d_NN_Denoiser(nn.Module):
    def __init__(self, kernel_size=3 ): 
        super(Conv1d_NN_Denoiser, self).__init__()
        self.kernel_size = kernel_size
        self.branch1 = BranchingNetwork(in_ch = 1, out_ch1 = 16, out_ch2=16, kernel_size = self.kernel_size)
        self.branch2 = BranchingNetwork(in_ch = 16, out_ch1 = 8, out_ch2=8, kernel_size = self.kernel_size)
        self.branch3 = BranchingNetwork(in_ch = 8, out_ch1 = 4, out_ch2=4, kernel_size = self.kernel_size)
        self.branch4 = BranchingNetwork(in_ch = 4, out_ch1 = 2, out_ch2=2, kernel_size = self.kernel_size)
        self.branch5 = BranchingNetwork(in_ch = 2, out_ch1 = 1, out_ch2=1, kernel_size = self.kernel_size)
        
    def count_params(self): 
        return sum([p.view(-1).shape[0] for p in self.parameters()])
    
    def forward(self, x):
        x = self.branch1(x)
        x = self.branch2(x)
        x = self.branch3(x)
        x = self.branch4(x)
        x = self.branch5(x)
        return x
