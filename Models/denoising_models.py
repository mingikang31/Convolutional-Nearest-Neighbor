'''2D Models for Convolutional Neural Networks.'''
'''Classification & Denoising Model'''
### All Models are based on the CIFAR10 dataset dimensions. ###

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchsummary import summary

# ConvNN
import sys
sys.path.append('../Layers')
from Conv1d_NN import *
from Conv2d_NN import *

from Conv1d_NN_spatial import *
from Conv2d_NN_spatial import *

from ConvNN_CNN_Branching import *

from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D

# Data + Training
sys.path.append('../Data')
from CIFAR10 import *

sys.path.append('../Train')
from train2d import *

'''
Denoising Models

1. Classic CNN: K = 3

2. ConvNN 2D: K = 9, N = All Samples
3. ConvNN 2D Random Sampling: K = 9, N = 64 Samples
4. ConvNN 2D Spatial Sampling: K = 9, N = 8 (N^2) Samples

5. ConvNN 2D: K = 9, N = All Samples, Location Channels
6. ConvNN 2D Random Sampling: K = 9, N = 64, Location Channels
7. ConvNN 2D Spatial Sampling: K = 9, N = 8 (N^2) Samples, Location Channels

8. Local -> Global ConvNN 2D: kernel_size = 3, K = 9, All Samples
9. Global -> Local ConvNN 2D: kernel_size = 3, K = 9, All Samples

10. Branching Network (ConvNN All Sample): kernel_size = 3, K = 9, N = All Samples
11. Branching Network (ConvNN Random Sample): kernel_size = 3, K = 9, N = 64 Samples
12. Branching Network (ConvNN Spatial Sample): kernel_size = 3, K = 9, N = 8 (N^2) Samples
'''

class DenoisingCNN(nn.Module):
    def __init__(self, in_ch=3, kernel_size=3):
        super(DenoisingCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Output activation
        
        self.to("mps")
        self.name = "DenoisingCNN"

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        # x = self.sigmoid(x)  # Apply sigmoid
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to("mps")

class DenoisingConvNN_2D_K_All(nn.Module):
    def __init__(self, in_ch=3, K=9):
        super(DenoisingConvNN_2D_K_All, self).__init__()
        self.conv1 = Conv2d_NN(in_ch, 16, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all")
        self.conv2 = Conv2d_NN(16, 32, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all")
        self.conv3 = Conv2d_NN(32, 3, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all") 

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.to("mps")
        self.name = "DenoisingConvNN_2D_K_All"


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        # x = self.sigmoid(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to("mps")

class DenoisingConvNN_2D_K_N(nn.Module):
    def __init__(self, in_ch=3, K=9, N = 64):
        super(DenoisingConvNN_2D_K_N, self).__init__()
        self.conv1 = Conv2d_NN(in_ch, 16, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples=N)
        self.conv2 = Conv2d_NN(16, 32, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples=N)
        self.conv3 = Conv2d_NN(32, 3, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples=N)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.to("mps")
        self.name = "DenoisingConvNN_2D_K_N"


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        # x = self.sigmoid(x)
        return x

    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to("mps")

class DenoisingConvNN_2D_Spatial_K_N(nn.Module):
    def __init__(self, in_ch=3, K=9, N = 8):
        super(DenoisingConvNN_2D_Spatial_K_N, self).__init__()
        self.conv1 = Conv2d_NN_spatial(in_ch, 16, K=K, stride=9, shuffle_pattern="BA", shuffle_scale=2, samples=N)
        self.conv2 = Conv2d_NN_spatial(16, 32, K=K, stride=9, shuffle_pattern="BA", shuffle_scale=2, samples=N)
        self.conv3 = Conv2d_NN_spatial(32, 3, K=K, stride=9, shuffle_pattern="BA", shuffle_scale=2, samples=N)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.to("mps")
        self.name = "DenoisingConvNN_2D_Spatial_K_N"


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        # x = self.sigmoid(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to("mps")

class DenoisingConvNN_2D_K_All_Location(nn.Module):
    def __init__(self, in_ch=3, K=9):
        super(DenoisingConvNN_2D_K_All_Location, self).__init__()
        self.conv1 = Conv2d_NN(in_ch, 16, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all", location_channels=True)
        self.conv2 = Conv2d_NN(16, 32, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all", location_channels=True)
        self.conv3 = Conv2d_NN(32, 3, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all", location_channels=True)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.to("mps")
        self.name = "DenoisingConvNN_2D_K_All_Location"


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        # x = self.sigmoid(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to("mps")
        
class DenoisingConvNN_2D_K_N_Location(nn.Module):
    def __init__(self, in_ch=3, K=9, N = 64):
        super(DenoisingConvNN_2D_K_N_Location, self).__init__()
        self.conv1 = Conv2d_NN(in_ch, 16, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples=N, location_channels=True)
        self.conv2 = Conv2d_NN(16, 32, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples=N, location_channels=True)
        self.conv3 = Conv2d_NN(32, 3, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples=N, location_channels=True)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.to("mps")
        self.name = "DenoisingConvNN_2D_K_N_Location"


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        # x = self.sigmoid(x)
        return x

    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to("mps")

class DenoisingConvNN_2D_Spatial_K_N_Location(nn.Module):
    def __init__(self, in_ch=3, K=9, N = 8):
        super(DenoisingConvNN_2D_Spatial_K_N_Location, self).__init__()
        self.conv1 = Conv2d_NN_spatial(in_ch, 16, K=K, stride=9, shuffle_pattern="BA", shuffle_scale=2, samples=N, location_channels=True)
        self.conv2 = Conv2d_NN_spatial(16, 32, K=K, stride=9, shuffle_pattern="BA", shuffle_scale=2, samples=N, location_channels=True)
        self.conv3 = Conv2d_NN_spatial(32, 3, K=K, stride=9, shuffle_pattern="BA", shuffle_scale=2, samples=N, location_channels=True)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.to("mps")
        self.name = "DenoisingConvNN_2D_Spatial_K_N_Location"


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        # x = self.sigmoid(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to("mps")

class DenoisingLocal_Global_ConvNN_2D(nn.Module):
    def __init__(self, in_ch=3, kernel_size=3, K=9, N = "all", location_channels=False):
        super(DenoisingLocal_Global_ConvNN_2D, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=kernel_size, stride=1, padding=1)
        self.conv2 = Conv2d_NN(16, 32, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples=N, location_channels=location_channels)
        self.conv3 = Conv2d_NN(32, 3, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples=N, location_channels=location_channels)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.to("mps")
        self.name = "DenoisingLocal_Global_ConvNN_2D"
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        # x = self.sigmoid(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to("mps")

class DenoisingGlobal_Local_ConvNN_2D(nn.Module):
    def __init__(self, in_ch=3, kernel_size=3, K=9, N = "all", location_channels=False):
        super(DenoisingGlobal_Local_ConvNN_2D, self).__init__()
        self.conv1 = Conv2d_NN(in_ch, 16, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples=N, location_channels=location_channels)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=kernel_size, stride=1, padding=1) 

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.to("mps")
        self.name = "DenoisingGlobal_Local_ConvNN_2D"

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        # x = self.sigmoid(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to("mps")
        
class DenoisingBranching_ConvNN_2D_K_All(nn.Module):
    def __init__(self, in_ch=3, channel_ratio=(16, 16), kernel_size=3, K=9, location_channels=False):
        super(DenoisingBranching_ConvNN_2D_K_All, self).__init__()
        self.conv1 = ConvNN_CNN_Random_BranchingLayer(in_ch, 16, channel_ratio=channel_ratio, kernel_size=kernel_size, K=K, location_channels=location_channels)
        self.conv2 = ConvNN_CNN_Random_BranchingLayer(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2),kernel_size=kernel_size, K=K, location_channels=location_channels)
        self.conv3 = ConvNN_CNN_Random_BranchingLayer(32, 3, channel_ratio=(3, 3), kernel_size=kernel_size, K=K, location_channels=location_channels)

        self.sigmoid = nn.Sigmoid()
        self.to("mps")
        self.name = "DenoisingBranching_ConvNN_2D_K_All"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.sigmoid(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to("mps")
        
        
class DenoisingBranching_ConvNN_2D_K_N(nn.Module):
    def __init__(self, in_ch=3, channel_ratio=(16, 16), kernel_size=3, K=9, N = 64, location_channels=False):
        super(DenoisingBranching_ConvNN_2D_K_N, self).__init__()
        self.conv1 = ConvNN_CNN_Random_BranchingLayer(in_ch, 16, channel_ratio=channel_ratio, kernel_size=kernel_size, K=K, samples=N, location_channels=location_channels)
        self.conv2 = ConvNN_CNN_Random_BranchingLayer(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2),kernel_size=kernel_size, K=K, samples=N, location_channels=location_channels)
        self.conv3 = ConvNN_CNN_Random_BranchingLayer(32, 3, channel_ratio=(3, 3), kernel_size=kernel_size, K=K, samples=N, location_channels=location_channels)

        self.sigmoid = nn.Sigmoid()
        self.to("mps")
        self.name = "DenoisingBranching_ConvNN_2D_K_N"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.sigmoid(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to("mps")
        
class DenoisingBranching_ConvNN_2D_Spatial_K_N(nn.Module):
    def __init__(self, in_ch=3, channel_ratio=(16, 16), kernel_size=3, K=9, N = 8, location_channels=False):
        super(DenoisingBranching_ConvNN_2D_Spatial_K_N, self).__init__()
        self.conv1 = ConvNN_CNN_Spatial_BranchingLayer(in_ch, 16, channel_ratio=channel_ratio, kernel_size=kernel_size, K=K, samples=N, location_channels=location_channels)
        self.conv2 = ConvNN_CNN_Spatial_BranchingLayer(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2),kernel_size=kernel_size, K=K, samples=N, location_channels=location_channels)
        self.conv3 = ConvNN_CNN_Spatial_BranchingLayer(32, 3, channel_ratio=(3, 3),kernel_size=kernel_size, K=K, samples=N, location_channels=location_channels)

        self.sigmoid = nn.Sigmoid()
        self.to("mps")
        self.name = "DenoisingBranching_ConvNN_2D_Spatial_K_N"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.sigmoid(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to("mps")

def denoising_check():
    
    # Models
    models = [DenoisingCNN(), DenoisingConvNN_2D_K_All(),
              DenoisingConvNN_2D_K_N(), DenoisingConvNN_2D_Spatial_K_N(),
              DenoisingConvNN_2D_K_All_Location(), DenoisingConvNN_2D_K_N_Location(),
              DenoisingConvNN_2D_Spatial_K_N_Location(), DenoisingLocal_Global_ConvNN_2D(),
              DenoisingGlobal_Local_ConvNN_2D(), 
              
              DenoisingBranching_ConvNN_2D_K_All(), DenoisingBranching_ConvNN_2D_K_N(),
              DenoisingBranching_ConvNN_2D_Spatial_K_N()
              
              ]

    # Data
    ex = torch.rand(1, 3, 32, 32).to("mps")
    
    # Training
    for model in models:
        try:
            ex_out = model(ex)
            print(f"Output Shape: {ex_out.shape}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == '__main__':
    
    # print("Denoising Models")
    # denoising_check()
    
    pass
    
    

    