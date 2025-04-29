'''2D Models for Convolutional Neural Networks.'''
'''Classification Model'''
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
from Conv2d_NN import Conv2d_NN
from Conv2d_NN_spatial import Conv2d_NN_spatial
from Conv2d_NN_Attn import Conv2d_NN_Attn
from Conv2d_NN_Attn_spatial import Conv2d_NN_Attn_spatial
from Conv2d_NN_Attn_V import Conv2d_NN_Attn_V
from Attention2d import Attention2d


from Attention_ConvNN_Branching import Attention_ConvNN_Random_Branching, Attention_ConvNN_Spatial_Branching, Attention_ConvNN_Attn_Branching, Attention_ConvNN_Attn_Spatial_Branching , Attention_ConvNN_Attn_V_Branching, Attention_Conv2d_Branching

from Conv2d_ConvNN_Branching import Conv2d_ConvNN_Random_Branching, Conv2d_ConvNN_Spatial_Branching, Conv2d_ConvNN_Attn_Branching, Conv2d_ConvNN_Attn_Spatial_Branching, Conv2d_ConvNN_Attn_V_Branching


class CNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, kernel_size=3, device="mps"):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "CNN"

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
        
class ConvNN_2D_K_All(nn.Module):
    def __init__(self, 
                 in_ch=3, 
                 K=9,
                 num_classes=10, shuffle_scale=2, shuffle_pattern="BA", 
                 device="mps"):
        super(ConvNN_2D_K_All, self).__init__()
        self.conv1 = Conv2d_NN(in_ch, 16, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples="all")
        self.conv2 = Conv2d_NN(16, 32, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples="all")

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "ConvNN_2D_K_All"

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
        
class ConvNN_2D_K_N(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, K=9, N=64, shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        super(ConvNN_2D_K_N, self).__init__()
        self.conv1 = Conv2d_NN(in_ch, 16, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=N)
        self.conv2 = Conv2d_NN(16, 32, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=N)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "ConvNN_2D_K_N"

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)

class ConvNN_2D_Spatial_K_N(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, K=9, N=8, shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        super(ConvNN_2D_Spatial_K_N, self).__init__()
        self.conv1 = Conv2d_NN_spatial(in_ch, 16, K=K, stride=9, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=N)
        self.conv2 = Conv2d_NN_spatial(16, 32, K=K, stride=9, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=N)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "ConvNN_2D_Spatial_K_N"

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)

### Location Models ###
class ConvNN_2D_K_All_Location(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, K=9, shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        super(ConvNN_2D_K_All_Location, self).__init__()
        self.conv1 = Conv2d_NN(in_ch, 16, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples="all", location_channels=True)
        self.conv2 = Conv2d_NN(16, 32, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples="all", location_channels=True)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "ConvNN_2D_K_All_Location"

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)

class ConvNN_2D_K_N_Location(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, K=9, N=64, shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        super(ConvNN_2D_K_N_Location, self).__init__()
        self.conv1 = Conv2d_NN(in_ch, 16, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=N, location_channels=True)
        self.conv2 = Conv2d_NN(16, 32, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=N, location_channels=True)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "ConvNN_2D_K_N_Location"

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)

class ConvNN_2D_Spatial_K_N_Location(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, K=9, N = 8, shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        super(ConvNN_2D_Spatial_K_N_Location, self).__init__()
        self.conv1 = Conv2d_NN_spatial(in_ch, 16, K=K, stride=9, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=N, location_channels=True)
        self.conv2 = Conv2d_NN_spatial(16, 32, K=K, stride=9, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=N, location_channels=True)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "ConvNN_2D_Spatial_K_N_Location"


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)

### Attention Models ### 
class ConvNN_2D_Attn_K_All(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, K=9, image_size=(32, 32), shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        super(ConvNN_2D_Attn_K_All, self).__init__()
        self.conv1 = Conv2d_NN_Attn(in_ch, 16, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples="all", image_size=image_size)
        self.conv2 = Conv2d_NN_Attn(16, 32, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples="all", image_size=image_size)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "ConvNN_2D_Attn_K_All"


    def forward(self, x):
        x = self.relu(self.conv1(x))        
        x = self.relu(self.conv2(x))

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
        
class ConvNN_2D_Attn_K_N(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, K=9, N=64, image_size=(32, 32), shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        super(ConvNN_2D_Attn_K_N, self).__init__()
        self.conv1 = Conv2d_NN_Attn(in_ch, 16, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=N, image_size=image_size)
        self.conv2 = Conv2d_NN_Attn(16, 32, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=N, image_size=image_size)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "ConvNN_2D_Attn_K_N"


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
        
class ConvNN_2D_Attn_Spatial_K_N(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, K=9, N=8, image_size=(32, 32), shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        super(ConvNN_2D_Attn_Spatial_K_N, self).__init__()
        self.conv1 = Conv2d_NN_Attn_spatial(in_ch, 16, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=N, image_size=image_size)
        self.conv2 = Conv2d_NN_Attn_spatial(16, 32, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=N, image_size=image_size)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "ConvNN_2D_Attn_Spatial_K_N"


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
        
class ConvNN_2D_Attn_V_K_All(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, K=9, image_size=(32, 32), shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        super(ConvNN_2D_Attn_V_K_All, self).__init__()
        self.conv1 = Conv2d_NN_Attn_V(in_ch, 16, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples="all", image_size=image_size)
        self.conv2 = Conv2d_NN_Attn_V(16, 32, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples="all", image_size=image_size)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "ConvNN_2D_Attn_V_K_All"


    def forward(self, x):
        x = self.relu(self.conv1(x))        
        x = self.relu(self.conv2(x))

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
        
class ConvNN_2D_Attn_V_K_N(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, K=9, N=64, image_size=(32, 32), shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        super(ConvNN_2D_Attn_V_K_N, self).__init__()
        self.conv1 = Conv2d_NN_Attn_V(in_ch, 16, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=N, image_size=image_size)
        self.conv2 = Conv2d_NN_Attn_V(16, 32, K=K, stride=K, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=N, image_size=image_size)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "ConvNN_2D_Attn_V_K_N"


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
            
### Attention Models ### 
class Attention_2D(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, num_heads=4, K=9, shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        super(Attention_2D, self).__init__()
        self.num_heads = num_heads
        
        self.conv1 = Attention2d(in_ch, 16, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, num_heads=self.num_heads)
        self.conv2 = Attention2d(16, 32, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, num_heads=self.num_heads)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "Attention_2D"


    def forward(self, x):
        x = self.relu(self.conv1(x))        
        x = self.relu(self.conv2(x))

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
        
### Local + Global Models ###
class Local_Global_ConvNN_2D(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, kernel_size=3, K=9, N = "all", location_channels = False, device="mps"):
        super(Local_Global_ConvNN_2D, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=kernel_size, stride=1, padding=1)
        self.conv2 = Conv2d_NN(16, 32, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples=N, location_channels=location_channels)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "Local_Global_ConvNN_2D"

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)

class Global_Local_ConvNN_2D(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, kernel_size=3, K=9, N = "all", location_channels = False, device="mps"):
        super(Global_Local_ConvNN_2D, self).__init__()
        self.conv1 = Conv2d_NN(in_ch, 16, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2, samples=N, location_channels=location_channels)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "Global_Local_ConvNN_2D"


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
        
### Branching Models ###
class B_Conv2d_ConvNN_K_All(nn.Module):
    def __init__(self, in_ch=3, channel_ratio=(16, 16), num_classes=10, kernel_size=3, K=9, location_channels=False, shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        super(B_Conv2d_ConvNN_K_All, self).__init__()
        self.conv1 = Conv2d_ConvNN_Random_Branching(in_ch, 16, channel_ratio=channel_ratio, kernel_size=kernel_size, K=K, samples="all", shuffle_pattern="BA", shuffle_scale=2, location_channels=location_channels)
        self.conv2 = Conv2d_ConvNN_Random_Branching(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2), kernel_size=kernel_size, K=K, samples="all", shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, location_channels=location_channels)    
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "B_Conv2d_ConvNN_K_All"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
         
class B_Conv2d_ConvNN_K_N(nn.Module):
    def __init__(self, in_ch=3,channel_ratio=(16, 16), num_classes=10, kernel_size=3, K=9, N=64, location_channels=False, shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        super(B_Conv2d_ConvNN_K_N, self).__init__()
        self.conv1 = Conv2d_ConvNN_Random_Branching(in_ch, 16, channel_ratio=channel_ratio, kernel_size=kernel_size, K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale,location_channels=location_channels)
        self.conv2 = Conv2d_ConvNN_Random_Branching(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2), kernel_size=kernel_size, K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, location_channels=location_channels)

        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "B_Conv2d_ConvNN_K_N"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
        
class B_Conv2d_ConvNN_Spatial_K_N(nn.Module):
    def __init__(self, in_ch=3, channel_ratio=(16, 16), num_classes=10, kernel_size=3, K=9, N=8, location_channels=False, shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        
        super(B_Conv2d_ConvNN_Spatial_K_N, self).__init__()
        self.conv1 = Conv2d_ConvNN_Spatial_Branching(in_ch, 16, channel_ratio=channel_ratio,kernel_size=kernel_size, K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, location_channels=location_channels)
        self.conv2 = Conv2d_ConvNN_Spatial_Branching(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2),kernel_size=kernel_size, K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, location_channels=location_channels)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "B_Conv2d_ConvNN_Spatial_K_N"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
        
class B_Conv2d_ConvNN_Attn_K_N(nn.Module):
    def __init__(self, in_ch=3, channel_ratio=(16, 16), num_classes=10, kernel_size=3, K=9, N=64, location_channels=False, image_size=(32, 32), shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        
        super(B_Conv2d_ConvNN_Attn_K_N, self).__init__()
        self.conv1 = Conv2d_ConvNN_Attn_Branching(in_ch, 16, channel_ratio=channel_ratio,kernel_size=kernel_size, K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, location_channels=location_channels, image_size=image_size)
        self.conv2 = Conv2d_ConvNN_Attn_Branching(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2),kernel_size=kernel_size, K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, location_channels=location_channels, image_size=image_size)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "B_Conv2d_ConvNN_Attn_K_N"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
        
class B_Conv2d_ConvNN_Attn_Spatial_K_N(nn.Module):
    def __init__(self, in_ch=3, channel_ratio=(16, 16), num_classes=10, kernel_size=3, K=9, N=8, location_channels=False, image_size=(32, 32), shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        
        super(B_Conv2d_ConvNN_Attn_Spatial_K_N, self).__init__()
        self.conv1 = Conv2d_ConvNN_Attn_Spatial_Branching(in_ch, 16, channel_ratio=channel_ratio,kernel_size=kernel_size, K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, location_channels=location_channels, image_size=image_size)
        self.conv2 = Conv2d_ConvNN_Attn_Spatial_Branching(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2),kernel_size=kernel_size, K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, location_channels=location_channels, image_size=image_size)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "B_Conv2d_ConvNN_Attn_Spatial_K_N"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
        
class B_Conv2d_ConvNN_Attn_V_K_N(nn.Module):
    def __init__(self, in_ch=3, channel_ratio=(16, 16), num_classes=10, kernel_size=3, K=9, N=64, location_channels=False, image_size=(32, 32), shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        
        super(B_Conv2d_ConvNN_Attn_V_K_N, self).__init__()
        self.conv1 = Conv2d_ConvNN_Attn_V_Branching(in_ch, 16, channel_ratio=channel_ratio,kernel_size=kernel_size, K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, location_channels=location_channels, image_size=image_size)
        self.conv2 = Conv2d_ConvNN_Attn_V_Branching(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2),kernel_size=kernel_size, K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, location_channels=location_channels, image_size=image_size)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "B_Conv2d_ConvNN_Attn_V_K_N"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)

class B_Attention_ConvNN_K_All(nn.Module):
    def __init__(self, in_ch=3, channel_ratio=(16, 16), num_classes=10, K=9, num_heads=4, location_channels = False, shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        
        super(B_Attention_ConvNN_K_All, self).__init__()
        self.conv1 = Attention_ConvNN_Random_Branching(in_ch, 16,  channel_ratio=channel_ratio, K=K, samples="all", shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, num_heads=num_heads, location_channels=location_channels)
        self.conv2 = Attention_ConvNN_Random_Branching(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2), K=K, samples="all", shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, num_heads=num_heads, location_channels=location_channels)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "B_Attention_ConvNN_K_All"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
        
class B_Attention_ConvNN_K_N(nn.Module):
    def __init__(self, in_ch=3, channel_ratio=(16, 16), num_classes=10, K=9, N=64, num_heads=4, location_channels=False, shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        
        super(B_Attention_ConvNN_K_N, self).__init__()
        self.conv1 = Attention_ConvNN_Random_Branching(in_ch, 16,  channel_ratio=channel_ratio, K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, num_heads=num_heads, location_channels=location_channels)
        self.conv2 = Attention_ConvNN_Random_Branching(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2), K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, num_heads=num_heads, location_channels=location_channels)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "B_Attention_ConvNN_K_N"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)

class B_Attention_ConvNN_Spatial_K_N(nn.Module):
    def __init__(self, in_ch=3, channel_ratio=(16, 16), num_classes=10, K=9, N=8, num_heads=4, location_channels=False, shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        
        super(B_Attention_ConvNN_Spatial_K_N, self).__init__()
        self.conv1 = Attention_ConvNN_Spatial_Branching(in_ch, 16,  channel_ratio=channel_ratio, K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, num_heads=num_heads, location_channels=location_channels)
        self.conv2 = Attention_ConvNN_Spatial_Branching(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2), K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, num_heads=num_heads, location_channels=location_channels)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "B_Attention_ConvNN_Spatial_K_N"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
        
class B_Attention_ConvNN_Attn_K_N(nn.Module):
    def __init__(self, in_ch=3, channel_ratio=(16, 16), num_classes=10, K=9, N=64, num_heads=4, image_size=(32, 32), location_channels=False, shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        
        super(B_Attention_ConvNN_Attn_K_N, self).__init__()
        self.conv1 = Attention_ConvNN_Attn_Branching(in_ch, 16,  channel_ratio=channel_ratio, K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, num_heads=num_heads, image_size=image_size, location_channels=location_channels)
        self.conv2 = Attention_ConvNN_Attn_Branching(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2), K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, num_heads=num_heads, image_size=image_size, location_channels=location_channels)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "B_Attention_ConvNN_Attn_K_N"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
        
class B_Attention_ConvNN_Attn_Spatial_K_N(nn.Module):
    def __init__(self, in_ch=3, channel_ratio=(16, 16), num_classes=10, K=9, N=8, num_heads=4, image_size=(32, 32), location_channels=False, shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        
        super(B_Attention_ConvNN_Attn_Spatial_K_N, self).__init__()
        self.conv1 = Attention_ConvNN_Attn_Spatial_Branching(in_ch, 16,  channel_ratio=channel_ratio, K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, num_heads=num_heads, image_size=image_size, location_channels=location_channels)
        self.conv2 = Attention_ConvNN_Attn_Spatial_Branching(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2), K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, num_heads=num_heads, image_size=image_size, location_channels=location_channels)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "B_Attention_ConvNN_Attn_Spatial_K_N"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)
         
class B_Attention_ConvNN_Attn_V_K_N(nn.Module):
    def __init__(self, in_ch=3, channel_ratio=(16, 16), num_classes=10, K=9, N=64, num_heads=4, image_size=(32, 32), location_channels=False, shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        
        super(B_Attention_ConvNN_Attn_V_K_N, self).__init__()
        self.conv1 = Attention_ConvNN_Attn_V_Branching(in_ch, 16,  channel_ratio=channel_ratio, K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, num_heads=num_heads, image_size=image_size, location_channels=location_channels)
        self.conv2 = Attention_ConvNN_Attn_V_Branching(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2), K=K, samples=N, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, num_heads=num_heads, image_size=image_size, location_channels=location_channels)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "B_Attention_ConvNN_Attn_V_K_N"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)

class B_Attention_Conv2d(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, channel_ratio=(16, 16),kernel_size=3, num_heads=4, shuffle_scale=2, shuffle_pattern="BA", device="mps"):
        
        super(B_Attention_Conv2d, self).__init__()
        self.conv1 = Attention_Conv2d_Branching(in_ch=in_ch, out_ch=16, channel_ratio=channel_ratio, kernel_size=kernel_size, num_heads=num_heads, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale)
        self.conv2 = Attention_Conv2d_Branching(in_ch=16, out_ch=32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2), kernel_size=kernel_size, num_heads=num_heads, shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.name = "B_Attention_Conv2d"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        self.to("cpu")
        print(summary(self, input_size))
        self.to(self.device)


### Location added before layers Models ** X' ###
# class CNN_Location_Before(nn.Module):
#     def __init__(self, in_ch=3, num_classes=10, kernel_size=3, device="mps"):
#         super(CNN_Location_Before, self).__init__()
        
        
#         self.conv1 = nn.Conv2d(in_ch+2, 16, kernel_size=kernel_size, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=1)

#         self.flatten = nn.Flatten()

#         self.fc1 = nn.Linear(32768, 1024)
#         self.fc2 = nn.Linear(1024, num_classes)
        
#         self.relu = nn.ReLU()
#         self.device = device
#         self.to(self.device)
#         self.name = "CNN_Location_Before"

#     def forward(self, x):
#         x_coordinates = self.coordinate_channels(x.shape, x.device)
#         x = torch.cat((x, x_coordinates), dim=1)
        
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.flatten(x)

#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)

#         return x
    
#     def summary(self, input_size = (3, 32, 32)): 
#         self.to("cpu")
#         print(summary(self, input_size))
#         self.to(self.device)
            
#     def coordinate_channels(self, tensor_shape, device):
#         x_ind = torch.arange(0, tensor_shape[2])
#         y_ind = torch.arange(0, tensor_shape[3])
        
#         x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
        
#         x_grid = x_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
#         y_grid = y_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        
#         xy_grid = torch.cat((x_grid, y_grid), dim=1)
#         xy_grid_normalized = F.normalize(xy_grid, p=2, dim=1)
#         return xy_grid_normalized.to(device)

# class ConvNN_2D_K_All_Location_Before(nn.Module):
#     def __init__(self, in_ch=3, num_classes=10, K=9, device="mps"):
#         super(ConvNN_2D_K_All_Location_Before, self).__init__()
        
#         self.conv1 = Conv2d_NN(in_ch+2, 16, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2,  samples="all")
#         self.conv2 = Conv2d_NN(16, 32, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2,  samples="all")

#         self.flatten = nn.Flatten()

#         self.fc1 = nn.Linear(32768, 1024)
#         self.fc2 = nn.Linear(1024, num_classes)

#         self.relu = nn.ReLU()
#         self.device = device
#         self.to(self.device)
#         self.name = "ConvNN_2D_K_All_Location_Before"

#     def forward(self, x):
#         x_coordinates = self.coordinate_channels(x.shape, x.device)
#         x = torch.cat((x, x_coordinates), dim=1)
        
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.flatten(x)

#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)

#         return x
    
#     def summary(self, input_size = (3, 32, 32)): 
#         self.to("cpu")
#         print(summary(self, input_size))
#         self.to(self.device)
        
#     def coordinate_channels(self, tensor_shape, device):
#         x_ind = torch.arange(0, tensor_shape[2])
#         y_ind = torch.arange(0, tensor_shape[3])
        
#         x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
        
#         x_grid = x_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
#         y_grid = y_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        
#         xy_grid = torch.cat((x_grid, y_grid), dim=1)
#         xy_grid_normalized = F.normalize(xy_grid, p=2, dim=1)
#         return xy_grid_normalized.to(device)

# class ConvNN_2D_K_N_Location_Before(nn.Module):
#     def __init__(self, in_ch=3, num_classes=10, K=9, N = 64, device="mps"):
#         super(ConvNN_2D_K_N_Location_Before, self).__init__()
        
#         self.conv1 = Conv2d_NN(in_ch+2, 16, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2,  samples=N)
#         self.conv2 = Conv2d_NN(16, 32, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2,  samples=N)

#         self.flatten = nn.Flatten()

#         self.fc1 = nn.Linear(32768, 1024)
#         self.fc2 = nn.Linear(1024, num_classes)

#         self.relu = nn.ReLU()
#         self.device = device
#         self.to(self.device)
#         self.name = "ConvNN_2D_K_N_Location_Before"

#     def forward(self, x):
#         x_coordinates = self.coordinate_channels(x.shape, x.device)
#         x = torch.cat((x, x_coordinates), dim=1)
        
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.flatten(x)

#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)

#         return x
    
#     def summary(self, input_size = (3, 32, 32)): 
#         self.to("cpu")
#         print(summary(self, input_size))
#         self.to(self.device)
        
#     def coordinate_channels(self, tensor_shape, device):
#         x_ind = torch.arange(0, tensor_shape[2])
#         y_ind = torch.arange(0, tensor_shape[3])
        
#         x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
        
#         x_grid = x_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
#         y_grid = y_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        
#         xy_grid = torch.cat((x_grid, y_grid), dim=1)
#         xy_grid_normalized = F.normalize(xy_grid, p=2, dim=1)
#         return xy_grid_normalized.to(device)

# class ConvNN_2D_Spatial_K_N_Location_Before(nn.Module):
#     def __init__(self, in_ch=3, num_classes=10, K=9, N = 8, device="mps"):
#         super(ConvNN_2D_Spatial_K_N_Location_Before, self).__init__()
        
#         self.conv1 = Conv2d_NN_spatial(in_ch+2, 16, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2,  samples=N)
#         self.conv2 = Conv2d_NN_spatial(16, 32, K=K, stride=K, shuffle_pattern="BA", shuffle_scale=2,  samples=N)

#         self.flatten = nn.Flatten()

#         self.fc1 = nn.Linear(32768, 1024)
#         self.fc2 = nn.Linear(1024, num_classes)

#         self.relu = nn.ReLU()
#         self.device = device
#         self.to(self.device)
#         self.name = "ConvNN_2D_Spatial_K_N_Location_Before"

#     def forward(self, x):
#         x_coordinates = self.coordinate_channels(x.shape, x.device)
#         x = torch.cat((x, x_coordinates), dim=1)
        
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.flatten(x)

#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)

#         return x
    
#     def summary(self, input_size = (3, 32, 32)): 
#         self.to("cpu")
#         print(summary(self, input_size))
#         self.to(self.device)
        
#     def coordinate_channels(self, tensor_shape, device):
#         x_ind = torch.arange(0, tensor_shape[2])
#         y_ind = torch.arange(0, tensor_shape[3])
        
#         x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
        
#         x_grid = x_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
#         y_grid = y_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        
#         xy_grid = torch.cat((x_grid, y_grid), dim=1)
#         xy_grid_normalized = F.normalize(xy_grid, p=2, dim=1)
#         return xy_grid_normalized.to(device)

# class Branching_ConvNN_2D_K_All_Location_Before(nn.Module):
#     def __init__(self, in_ch=3, channel_ratio=(16, 16), num_classes=10, kernel_size=3, K=9, location_channels = False, device="mps"):
        
#         super(Branching_ConvNN_2D_K_All_Location_Before, self).__init__()
#         self.conv1 = ConvNN_CNN_Random_BranchingLayer(in_ch+2, 16, 
#             channel_ratio=channel_ratio,kernel_size=kernel_size, K=K, samples="all", shuffle_pattern="BA", shuffle_scale=2, location_channels=location_channels)
#         self.conv2 = ConvNN_CNN_Random_BranchingLayer(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2),kernel_size=kernel_size, K=K, samples="all", shuffle_pattern="BA", shuffle_scale=2, location_channels=location_channels)
        
#         self.flatten = nn.Flatten()

#         self.fc1 = nn.Linear(32768, 1024)
#         self.fc2 = nn.Linear(1024, num_classes)

#         self.relu = nn.ReLU()
#         self.device = device
#         self.to(self.device)
#         self.name = "Branching_ConvNN_2D_K_All_Location_Before"

#     def forward(self, x):
#         x_coordinates = self.coordinate_channels(x.shape, x.device)
#         x = torch.cat((x, x_coordinates), dim=1)
#         x = self.conv1(x)
#         x = self.conv2(x)
        
#         x = self.flatten(x)

#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
#     def summary(self, input_size = (3, 32, 32)): 
#         self.to("cpu")
#         print(summary(self, input_size))
#         self.to(self.device)
        
#     def coordinate_channels(self, tensor_shape, device):
#         x_ind = torch.arange(0, tensor_shape[2])
#         y_ind = torch.arange(0, tensor_shape[3])
        
#         x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
        
#         x_grid = x_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
#         y_grid = y_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        
#         xy_grid = torch.cat((x_grid, y_grid), dim=1)
#         xy_grid_normalized = F.normalize(xy_grid, p=2, dim=1)
#         return xy_grid_normalized.to(device)
    
# class Branching_ConvNN_2D_K_N_Location_Before(nn.Module):
#     def __init__(self, in_ch=3, channel_ratio=(16, 16), num_classes=10, kernel_size=3, K=9, N = 64, location_channels = False, device="mps"):
        
#         super(Branching_ConvNN_2D_K_N_Location_Before, self).__init__()
#         self.conv1 = ConvNN_CNN_Random_BranchingLayer(in_ch+2, 16, 
#             channel_ratio=channel_ratio,kernel_size=kernel_size, K=K, samples=N, shuffle_pattern="BA", shuffle_scale=2, location_channels=location_channels)
#         self.conv2 = ConvNN_CNN_Random_BranchingLayer(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2),kernel_size=kernel_size, K=K, samples=N, shuffle_pattern="BA", shuffle_scale=2, location_channels=location_channels)
        
#         self.flatten = nn.Flatten()

#         self.fc1 = nn.Linear(32768, 1024)
#         self.fc2 = nn.Linear(1024, num_classes)

#         self.relu = nn.ReLU()
#         self.device = device
#         self.to(self.device)
#         self.name = "Branching_ConvNN_2D_K_N_Location_Before"

#     def forward(self, x):
#         x_coordinates = self.coordinate_channels(x.shape, x.device)
#         x = torch.cat((x, x_coordinates), dim=1)
#         x = self.conv1(x)
#         x = self.conv2(x)
        
#         x = self.flatten(x)

#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
#     def summary(self, input_size = (3, 32, 32)): 
#         self.to("cpu")
#         print(summary(self, input_size))
#         self.to(self.device)
        
#     def coordinate_channels(self, tensor_shape, device):
#         x_ind = torch.arange(0, tensor_shape[2])
#         y_ind = torch.arange(0, tensor_shape[3])
        
#         x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
        
#         x_grid = x_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
#         y_grid = y_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        
#         xy_grid = torch.cat((x_grid, y_grid), dim=1)
#         xy_grid_normalized = F.normalize(xy_grid, p=2, dim=1)
#         return xy_grid_normalized.to(device)

# class Branching_ConvNN_2D_Spatial_K_N_Location_Before(nn.Module):
#     def __init__(self, in_ch=3, channel_ratio=(16, 16), num_classes=10, kernel_size=3, K=9, N = 8, location_channels = False, device="mps"):
        
#         super(Branching_ConvNN_2D_Spatial_K_N_Location_Before, self).__init__()
#         self.conv1 = ConvNN_CNN_Spatial_BranchingLayer(in_ch+2, 16, 
#             channel_ratio=channel_ratio,kernel_size=kernel_size, K=K, samples=N, shuffle_pattern="BA", shuffle_scale=2, location_channels=location_channels)
#         self.conv2 = ConvNN_CNN_Spatial_BranchingLayer(16, 32, channel_ratio=(channel_ratio[0] *2, channel_ratio[1]*2),kernel_size=kernel_size, K=K, samples=N, shuffle_pattern="BA", shuffle_scale=2, location_channels=location_channels)
        
#         self.flatten = nn.Flatten()

#         self.fc1 = nn.Linear(32768, 1024)
#         self.fc2 = nn.Linear(1024, num_classes)

#         self.relu = nn.ReLU()
#         self.device = device
#         self.to(self.device)
#         self.name = "Branching_ConvNN_2D_Spatial_K_N_Location_Before"

#     def forward(self, x):
#         x_coordinates = self.coordinate_channels(x.shape, x.device)
#         x = torch.cat((x, x_coordinates), dim=1)
#         x = self.conv1(x)
#         x = self.conv2(x)
        
#         x = self.flatten(x)

#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
#     def summary(self, input_size = (3, 32, 32)): 
#         self.to("cpu")
#         print(summary(self, input_size))
#         self.to(self.device)
        
#     def coordinate_channels(self, tensor_shape, device):
#         x_ind = torch.arange(0, tensor_shape[2])
#         y_ind = torch.arange(0, tensor_shape[3])
        
#         x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
        
#         x_grid = x_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
#         y_grid = y_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        
#         xy_grid = torch.cat((x_grid, y_grid), dim=1)
#         xy_grid_normalized = F.normalize(xy_grid, p=2, dim=1)
#         return xy_grid_normalized.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def classification_check():
    # Models
    models = [CNN(), 
              
                ConvNN_2D_K_All(), 
                ConvNN_2D_K_N(), ConvNN_2D_Spatial_K_N(),
                ConvNN_2D_K_All_Location(), ConvNN_2D_K_N_Location(), 
                ConvNN_2D_Spatial_K_N_Location(),

                ConvNN_2D_Attn_K_All(), 
                ConvNN_2D_Attn_K_N(), 
                ConvNN_2D_Attn_Spatial_K_N(),
                ConvNN_2D_Attn_V_K_All(),
                ConvNN_2D_Attn_V_K_N(),

                Attention_2D(),

                Local_Global_ConvNN_2D(), 
                Global_Local_ConvNN_2D(), 

                B_Conv2d_ConvNN_K_All(),
                B_Conv2d_ConvNN_K_N(),
                B_Conv2d_ConvNN_Spatial_K_N(),
                B_Conv2d_ConvNN_Attn_K_N(),
                B_Conv2d_ConvNN_Attn_Spatial_K_N(),
                B_Conv2d_ConvNN_Attn_V_K_N(),
                B_Attention_ConvNN_K_All(),
                B_Attention_ConvNN_K_N(),
                B_Attention_ConvNN_Spatial_K_N(),
                B_Attention_ConvNN_Attn_K_N(),
                B_Attention_ConvNN_Attn_Spatial_K_N(),
                B_Attention_ConvNN_Attn_V_K_N(),
                B_Attention_Conv2d(),
                


                # CNN_Location_Before(), ConvNN_2D_K_All_Location_Before(),
                # ConvNN_2D_K_N_Location_Before(), ConvNN_2D_Spatial_K_N_Location_Before(),
                # Branching_ConvNN_2D_K_All_Location_Before(), Branching_ConvNN_2D_K_N_Location_Before(),
                # Branching_ConvNN_2D_Spatial_K_N_Location_Before()
              ]
              
    # Data
    ex = torch.rand(1, 3, 32, 32).to("mps")

    # Testing
    for model in models:
        try:
            ex_out = model(ex)
            print(f"{model.name}'s output Shape: {ex_out.shape}\n")
            
            # print(f"{model.name}'s Number of Parameters: {count_parameters(model)}\n")
            
            # print(model.summary())
        except Exception as e:
            print(f"Error: {e}\n")
            

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == '__main__':
    
    print("Classification Models")
    classification_check()    
    
