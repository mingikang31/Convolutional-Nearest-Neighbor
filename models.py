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
            ### Implement Conv1d_NN
            # Unshuffle
            # Conv1d_NN(in_ch, out_ch2, K = kernel_size, stride = , sh_pat="BA", sh_scal),
            Conv1d_NN(in_ch, out_ch2, K = kernel_size, stride = kernel_size), 
            nn.ReLU()
            # Shuffle 
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
    

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(Conv1d_NN(in_channels=1, out_channels=16, K=5, stride=5), nn.ReLU())
        self.down1 = PixelUnshuffle1D(2)
        self.enc2 = nn.Sequential(Conv1d_NN(in_channels=32, out_channels=64, K=5, stride=5), nn.ReLU())
        self.down2 = PixelUnshuffle1D(2)
        self.enc3 = nn.Sequential(Conv1d_NN(in_channels=128, out_channels=256, K=5, stride=5), nn.ReLU())
        self.down3 = PixelUnshuffle1D(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(Conv1d_NN(in_channels=512, out_channels=512, K=5, stride=5), nn.ReLU())
        
        # Decoder
        self.up1 = PixelShuffle1D(2)
        self.dec1 = nn.Sequential(Conv1d_NN(in_channels=256, out_channels=128, K=5, stride=5), nn.ReLU())
        self.up2 = PixelShuffle1D(2)
        self.dec2 = nn.Sequential(Conv1d_NN(in_channels=64, out_channels=32, K=5, stride=5), nn.ReLU())
        self.up3 = PixelShuffle1D(2)
        self.dec3 = nn.Sequential(Conv1d_NN(in_channels=16, out_channels=5, K=5, stride=5), nn.ReLU())

        
        # Final layer
        self.final = nn.Sequential(nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1), nn.Flatten(), nn.Linear(40, 10), nn.ReLU())

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)        
        down1 = self.down1(enc1)
        enc2 = self.enc2(down1)
        down2 = self.down2(enc2)
        enc3 = self.enc3(down2)
        down3 = self.down3(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(down3)
        
        # Decoder
        up1 = self.up1(bottleneck)
        dec1 = self.dec1(up1)
        up2 = self.up2(dec1)
        dec2 = self.dec2(up2)
        up3 = self.up3(dec2)
        dec3 = self.dec3(up3)
        # Final layer
        out = self.final(dec3)
        return out
    
'''Pixel Shuffle Models + U-Net'''
''' MUST FIX LATER
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
    
'''

'''Denoising Models'''
# Conv1d Denoiser
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
    
    
Branching_Denoiser = nn.Sequential(
   BranchingNetwork(in_ch = 1, out_ch1 = 16, out_ch2=16, kernel_size = 3), 
   BranchingNetwork(in_ch = 16, out_ch1 = 8, out_ch2=8, kernel_size = 3),
   BranchingNetwork(in_ch = 8, out_ch1 = 4, out_ch2=4, kernel_size =3), 
   BranchingNetwork(in_ch = 4, out_ch1 = 2, out_ch2=2, kernel_size =3), 
   BranchingNetwork(in_ch = 2, out_ch1 = 1, out_ch2=1, kernel_size =3) 
)


class UNet_Denoiser(nn.Module):
    def __init__(self):
        super(UNet_Denoiser, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(Conv1d_NN(in_channels=1, out_channels=16, K=5, stride=5), nn.ReLU())
        self.down1 = PixelUnshuffle1D(2)
        self.enc2 = nn.Sequential(Conv1d_NN(in_channels=32, out_channels=64, K=5, stride=5), nn.ReLU())
        self.down2 = PixelUnshuffle1D(2)
        self.enc3 = nn.Sequential(Conv1d_NN(in_channels=128, out_channels=256, K=5, stride=5), nn.ReLU())
        self.down3 = PixelUnshuffle1D(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(Conv1d_NN(in_channels=512, out_channels=512, K=5, stride=5), nn.ReLU())
        
        # Decoder
        self.up1 = PixelShuffle1D(2)
        self.dec1 = nn.Sequential(Conv1d_NN(in_channels=256, out_channels=128, K=5, stride=5), nn.ReLU())
        self.up2 = PixelShuffle1D(2)
        self.dec2 = nn.Sequential(Conv1d_NN(in_channels=64, out_channels=32, K=5, stride=5), nn.ReLU())
        self.up3 = PixelShuffle1D(2)
        self.dec3 = nn.Sequential(Conv1d_NN(in_channels=16, out_channels=5, K=5, stride=5), nn.ReLU())

        
        # Final layer
        self.final = nn.Sequential(nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1))

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)        
        down1 = self.down1(enc1)
        enc2 = self.enc2(down1)
        down2 = self.down2(enc2)
        enc3 = self.enc3(down2)
        down3 = self.down3(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(down3)
        
        # Decoder
        up1 = self.up1(bottleneck)
        dec1 = self.dec1(up1)
        up2 = self.up2(dec1)
        dec2 = self.dec2(up2)
        up3 = self.up3(dec2)
        dec3 = self.dec3(up3)
        # Final layer
        out = self.final(dec3)
        return out

    
