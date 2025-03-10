'''2D Models for Convolutional Neural Networks - Denoising Models'''
'''Denoising Models for BSD68 dataset'''
### All Models are based on the BSD68 dataset: [1, 100, 100]

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


class DenoisingCNN_BSD(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, kernel_size=3):
        super(DenoisingCNN_BSD, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size, stride=1, padding=1)  
        self.conv4 = nn.Conv2d(64, 32, kernel_size, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 16, kernel_size, stride=1, padding=1)
        self.conv6 = nn.Conv2d(16, out_ch, kernel_size, stride=1, padding=1)
        
        self.relu = nn.ReLU()
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        self.name = "DenoisingCNN_BSD"
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        return x

    def summary(self, input_size=(1, 100, 100)):
        self.to('cpu')
        summary(self, input_size=input_size)
        self.to(self.device)

class Denoising_ConvNN_2D_K_All_BSD(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, K=3):
        super(Denoising_ConvNN_2D_K_All_BSD, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.K = K

        self.conv1 = Conv2d_NN(in_ch, 16, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all")
        self.conv2 = Conv2d_NN(16, 32, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all")
        self.conv3 = Conv2d_NN(32, 64, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all")
        self.conv4 = Conv2d_NN(64, 32, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all")
        self.conv5 = Conv2d_NN(32, 16, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all")
        self.conv6 = Conv2d_NN(16, out_ch, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all")
        
        self.relu = nn.ReLU()
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        self.name = "Denoising_ConvNN_2D_K_All_BSD"
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        return x

    def summary(self, input_size=(1, 100, 100)):
        self.to('cpu')
        summary(self, input_size=input_size)
        self.to(self.device)



class Denoising_ConvNN_2D_K_All_Prev_BSD(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, K=3):
        super(Denoising_ConvNN_2D_K_All_Prev_BSD, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.K = K

        self.conv1 = Conv2d_NN_prev(in_ch, 16, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all")
        self.conv2 = Conv2d_NN_prev(16, 32, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all")
        self.conv3 = Conv2d_NN_prev(32, 64, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all")
        self.conv4 = Conv2d_NN_prev(64, 32, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all")
        self.conv5 = Conv2d_NN_prev(32, 16, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all")
        self.conv6 = Conv2d_NN_prev(16, out_ch, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all")
        
        self.relu = nn.ReLU()
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        self.name = "Denoising_ConvNN_2D_K_All_Prev_BSD"
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        return x

    def summary(self, input_size=(1, 100, 100)):
        self.to('cpu')
        summary(self, input_size=input_size)
        self.to(self.device)

        
def example_DenoisingCNN_BSD():
    models = [
        DenoisingCNN_BSD(),
        Denoising_ConvNN_2D_K_All_BSD(),
        Denoising_ConvNN_2D_K_All_Prev_BSD()
    ]
    
    ex = torch.randn(1, 1, 100, 100)
    
    for model in models:
        try:
            ex_out = model(ex)
            print(f"Output Shape: {ex_out.shape}\n")
        except Exception as e:
            print(f"Error: {e}\n")
        
