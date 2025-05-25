# Torch Imports 
import torch
import torch.nn as nn
import torch.nn.functional as F

# ConvNN Layers
from layers2d import (
    Conv2d_NN, 
    Conv2d_NN_Spatial, 
    Conv2d_NN_Attn, 
    Conv2d_NN_Attn_Spatial, 
    Attention2d
)
# Branching Layers: Conv2d + ConvNN
from layers2d import (
    Conv2d_ConvNN_Branching, 
    Conv2d_ConvNN_Spatial_Branching,
    Conv2d_ConvNN_Attn_Branching,
    Conv2d_ConvNN_Attn_Spatial_Branching
)
# Branching Layers: Attention + ConvNN
from layers2d import (
    Attention_ConvNN_Branching, 
    Attention_ConvNN_Spatial_Branching, 
    Attention_ConvNN_Attn_Branching,
    Attention_ConvNN_Attn_Spatial_Branching
)
# Branching Layers: Attention + Conv2d
from layers2d import (
    Attention_Conv2d_Branching
)


from functools import partial
from typing import Any, cast, Optional, Union

import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    if hasattr(m, 'weight') and m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

'''**** Had to change the samples within the layers due to the image size reducing ****'''
## Everytime it reduces, the number of samples should be reduced by a factor of 4

def make_layers(cfg: list[Union[str, int]], 
                conv_layer: nn.Module = nn.Conv2d,  
                shuffle_pattern: str = "BA", 
                shuffle_scale: int = 2, 
                samples="all", 
                magnitude_type="similarity", 
                location_channels=False, 
                batch_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    current_size = 224
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            current_size //= 2
        else:

            v = cast(int, v)
            if conv_layer == nn.Conv2d: 
                conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if conv_layer == Conv2d_NN:
                if samples == "all":
                    conv = Conv2d_NN(in_channels, v, K=9, stride=9, padding=0, 
                                    shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=samples,
                                    magnitude_type=magnitude_type, location_channels=location_channels)
                else: 
                    conv = Conv2d_NN(in_channels, v, K=9, stride=9, padding=0, 
                                    shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=int((current_size/4)**2),
                                    magnitude_type=magnitude_type, location_channels=location_channels)
            if conv_layer == Conv2d_NN_Spatial:
                conv = Conv2d_NN_Spatial(in_channels, v, K=9, stride=9, padding=0, 
                                         shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=int(current_size/4),
                                         sample_padding=0,
                                         magnitude_type=magnitude_type, location_channels=location_channels)
            if conv_layer == Conv2d_NN_Attn:
                if samples == "all":
                    conv = Conv2d_NN_Attn(in_channels, v, K=9, stride=9, padding=0, 
                                        shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples="all",
                                        image_size=(current_size, current_size),
                                        magnitude_type=magnitude_type, location_channels=location_channels)
                else: 
                    conv = Conv2d_NN_Attn(in_channels, v, K=9, stride=9, padding=0, 
                    shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=int((current_size/4)**2),
                    image_size=(current_size, current_size),
                    magnitude_type=magnitude_type, location_channels=location_channels)              
            if conv_layer == Conv2d_NN_Attn_Spatial:
                conv = Conv2d_NN_Attn_Spatial(in_channels, v, K=9, stride=9, padding=0,  
                                             shuffle_pattern=shuffle_pattern,shuffle_scale=shuffle_scale, samples=int(current_size/4),
                                             samples_padding=0,
                                             image_size=(current_size, current_size),
                                             magnitude_type=magnitude_type, location_channels=location_channels)
            if conv_layer == Conv2d_ConvNN_Branching:
                if samples == "all":
                    conv = Conv2d_ConvNN_Branching(in_channels, 
                                                v, 
                                                channel_ratio=(v, v),
                                                kernel_size=3, 
                                                K=9, 
                                                shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples="all",
                                                magnitude_type=magnitude_type, location_channels=location_channels)
                else:
                    conv = Conv2d_ConvNN_Branching(in_channels, 
                                                v, 
                                                channel_ratio=(v, v),
                                                kernel_size=3, 
                                                K=9, 
                                                shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=int((current_size/4)**2),
                                                magnitude_type=magnitude_type, location_channels=location_channels)
            if conv_layer == Conv2d_ConvNN_Spatial_Branching:
                conv = Conv2d_ConvNN_Spatial_Branching(in_channels, 
                                                       v, 
                                                       channel_ratio=(v, v),
                                                       kernel_size=3, 
                                                       K=9, 
                                                       shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=int(current_size/4),
                                                      magnitude_type=magnitude_type, location_channels=location_channels)
            if conv_layer == Conv2d_ConvNN_Attn_Branching:
                if samples == "all":
                    conv = Conv2d_ConvNN_Attn_Branching(in_channels, 
                                                    v, 
                                                    channel_ratio=(v, v),
                                                    kernel_size=3, 
                                                    K=9, 
                                                    shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples="all",
                                                    image_size=(current_size, current_size),
                                                    magnitude_type=magnitude_type, location_channels=location_channels)
                else:
                    conv = Conv2d_ConvNN_Attn_Branching(in_channels, 
                                                    v, 
                                                    channel_ratio=(v, v),
                                                    kernel_size=3, 
                                                    K=9, 
                                                    shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=int((current_size/4)**2),
                                                    image_size=(current_size, current_size),
                                                    magnitude_type=magnitude_type, location_channels=location_channels)
            if conv_layer == Conv2d_ConvNN_Attn_Spatial_Branching:
                conv = Conv2d_ConvNN_Attn_Spatial_Branching(in_channels, 
                                                           v, 
                                                           channel_ratio=(v, v),
                                                           kernel_size=3, 
                                                           K=9, 
                                                           shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=int(current_size/4),
                                                          image_size=(current_size, current_size),
                                                          magnitude_type=magnitude_type, location_channels=location_channels)
            if conv_layer == Attention_ConvNN_Branching:
                if samples == "all":
                    conv = Attention_ConvNN_Branching(in_channels, 
                                                    v, 
                                                    channel_ratio=(v, v),
                                                    num_heads=4, 
                                                    K=9, 
                                                    shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples="all",
                                                    magnitude_type=magnitude_type, location_channels=location_channels)
                else: 
                    conv = Attention_ConvNN_Branching(in_channels, 
                                                    v, 
                                                    channel_ratio=(v, v),
                                                    num_heads=4, 
                                                    K=9, 
                                                    shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=int((current_size/4)**2),
                                                    magnitude_type=magnitude_type, location_channels=location_channels)
            if conv_layer == Attention_ConvNN_Spatial_Branching:
                conv = Attention_ConvNN_Spatial_Branching(in_channels, 
                                                         v, 
                                                         channel_ratio=(v, v),
                                                         num_heads=4, 
                                                         K=9, 
                                                         shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=int(current_size/4),
                                                        magnitude_type=magnitude_type, location_channels=location_channels)
            if conv_layer == Attention_ConvNN_Attn_Branching:
                if samples == "all":
                    conv = Attention_ConvNN_Attn_Branching(in_channels, 
                                                        v, 
                                                        channel_ratio=(v, v),
                                                        num_heads=4, 
                                                        K=9, 
                                                        shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples="all",
                                                        image_size=(current_size, current_size),
                                                        magnitude_type=magnitude_type, location_channels=location_channels)
                else: 
                    conv = Attention_ConvNN_Attn_Branching(in_channels, 
                                                        v, 
                                                        channel_ratio=(v, v),
                                                        num_heads=4, 
                                                        K=9, 
                                                        shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=int((current_size/4)**2),
                                                        image_size=(current_size, current_size),
                                                        magnitude_type=magnitude_type, location_channels=location_channels)
            if conv_layer == Attention_ConvNN_Attn_Spatial_Branching:
                conv = Attention_ConvNN_Attn_Spatial_Branching(in_channels, 
                                                             v, 
                                                             channel_ratio=(v, v),
                                                             num_heads=4, 
                                                             K=9, 
                                                             shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=int(current_size/4),
                                                            image_size=(current_size, current_size),
                                                            magnitude_type=magnitude_type, location_channels=location_channels)
            if conv_layer == Attention_Conv2d_Branching:
                conv = Attention_Conv2d_Branching(in_channels, 
                                                  v, 
                                                  channel_ratio=(v, v),
                                                  num_heads=4, 
                                                  kernel_size=3, 
                                                  shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale,
                                                 location_channels=location_channels)
            
            
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: dict[str, list[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"], ## VGG16 Model 
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, 
         batch_norm: bool, 
         num_classes: int = 1000,
         init_weights: bool = True, dropout: float = 0.5,
         
         conv_layer=nn.Conv2d, 
         shuffle_pattern: str = "BA", 
         shuffle_scale: int = 2, 
         samples="all", 
         magnitude_type="similarity", 
         location_channels=False, 
         ) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, conv_layer=conv_layer, 
                            shuffle_pattern=shuffle_pattern, shuffle_scale=shuffle_scale, samples=samples,
                            magnitude_type=magnitude_type, location_channels=location_channels), num_classes=num_classes,
                init_weights=init_weights, dropout=dropout)
    
    return model

if __name__ == "__main__":
    # Example usage
    model = _vgg("D", batch_norm=False, num_classes=100, samples=1, conv_layer=Conv2d_NN_Attn)
    x = torch.randn(1, 3, 224, 224)  # Example input
    output = model(x)
    print(output.shape)  # Should be [1, 1000]
    
    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    

'''
batch_size = 128    
num_epochs = 250
lr = 1e-1
weight_decay = 1e-6
momentum = 0.9
lr_step = 20
gamma = 0.5
'''
