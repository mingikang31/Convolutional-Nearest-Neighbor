import torch
import torch.nn as nn
from functools import partial
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, VisionTransformer # Import VisionTransformer here
from torchsummary import summary

class Attention1d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 shuffle_pattern='N/A', 
                 shuffle_scale=1, 
                 num_heads=1
                 ):
        super(Attention1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.num_heads = num_heads
        
        
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        
        # Channels for Attention 
        self.in_channels = self.in_channels * self.shuffle_scale if self.shuffle_pattern in ["BA", "B"] else in_channels
        self.out_channels = self.out_channels * self.shuffle_scale if self.shuffle_pattern in ["BA", "A"] else out_channels
        
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=self.out_channels, num_heads=self.num_heads, batch_first=True)
        
        self.conv1x1 = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1)
        
    def forward(self, x):
        if self.shuffle_pattern in ["BA", "B"]:
            x1 = self.unshuffle_layer(x)
        else: 
            x1 = x 
        
        x1 = self.conv1x1(x1) # [B, C, N]
        x1 = x1.permute(0, 2, 1)
        
        x2 = self.multi_head_attention(x1, x1, x1)[0] # (B, N, C)
        x2 = x2.permute(0, 2, 1) # (B, C, N)
        
        if self.shuffle_pattern in ["BA", "A"]:
            x3 = self.shuffle_layer(x2)
        else: 
            x3 = x2
        return x3
    
class Attention2d(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 shuffle_pattern='N/A',
                 shuffle_scale=1,
                 num_heads=1,
                 location_channels=False,
                 ): 
        super(Attention2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.num_heads = num_heads
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
                
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        
        self.attention1d = Attention1d(in_channels=self.in_channels_1d,
                                        out_channels=self.out_channels_1d,
                                        shuffle_pattern="N/A",
                                        shuffle_scale=1,
                                        num_heads=self.num_heads
                                          )
        
        self.flatten = nn.Flatten(start_dim=2)
        self.pointwise_conv = nn.Conv2d(self.out_channels + 2, self.out_channels, kernel_size=1)
        
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
        x3 = self.attention1d(x2)

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
        x_ind = torch.arange(0, tensor_shape[2])
        y_ind = torch.arange(0, tensor_shape[3])
        
        x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
        
        x_grid = x_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        y_grid = y_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        
        xy_grid = torch.cat((x_grid, y_grid), dim=1)
        xy_grid_normalized = F.normalize(xy_grid, p=2, dim=1)
        return xy_grid_normalized.to(device)

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

class Attention(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16, num_layers=2, num_heads=4, num_classes=100, device="mps"):
        super(Attention, self).__init__()
        
        assert num_layers >= 2, "Number of layers must be at least 2"
        assert mid_ch >= 8, "Middle channels must be at least 8"


        self.in_ch = in_ch
        self.mid_ch = mid_ch 
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(Attention2d(in_ch, mid_ch, shuffle_pattern='BA', shuffle_scale=2, num_heads=self.num_heads))
                layers.append(nn.ReLU())
            else: 
                layers.append(Attention2d(mid_ch, mid_ch, shuffle_pattern='BA', shuffle_scale=2, num_heads=self.num_heads))                
                layers.append(nn.ReLU())
        
        self.features = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        
        flattened_size = mid_ch * 32 * 32
        
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(flattened_size, num_classes), # Increased intermediate size
        )

        self.device = device
        self.to(self.device)
        self.name = "Attention" # Renamed for clarity
    
    def forward(self, x):   
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    def summary(self, input_size = (3, 32, 32)):
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but input_size doesn't include it
            summary(self, input_size=input_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)
            
@register_model 
def attention_100(pretrained=False, **kwargs):
    model = Attention(in_ch=3, mid_ch=16, num_layers=2, num_heads=4, num_classes=100, device="mps")
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model 
def attention_10(pretrained=False, **kwargs):
    model = Attention(in_ch=3, mid_ch=16, num_layers=2, num_heads=4, num_classes=10, device="mps")
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Quick test and parameter count comparison
    models = [
    attention_100(), 
    attention_10(),
        
    ]

    x = torch.randn(1, 3, 32, 32).to('mps')

    for model in models:
        print(f"Model: {model.name}")
        print(f"Parameters: {count_parameters(model):,}")
        print(f"Output shape: {model(x).shape}")
        print("-" * 30)


