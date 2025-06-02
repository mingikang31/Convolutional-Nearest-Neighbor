'''VGG Model with Conv2d, ConvNN, ConvNN_Attn, Attention'''

# Torch Imports 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchsummary import summary 

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

from typing import cast, Union


'''VGG Model Class'''
class VGG(nn.Module): 
    def __init__(self, args): 
        
        super(VGG, self).__init__()
        self.args = args 
        self.layer = args.layer
        self.model = "VGG"
        
        
        # Model Parameters
        self.sampling = args.sampling
        self.samples = "all" if args.sampling == "All" else None
        self.shuffle_pattern = args.shuffle_pattern
        self.shuffle_scale = int(args.shuffle_scale)
        self.magnitude_type = args.magnitude_type
        self.location_channels = args.location_channels
        self.dropout = float(args.dropout)
        
        self.num_classes = int(args.num_classes)
        self.device = args.device
        
        # In Channels, Middle Channels, and Number of Layers
        self.img_size = args.img_size
        self.in_channels = int(self.img_size[0]) # Number of Channels
        
        # cfgs for VGG Model (VGG16, VGG19)
        cfgs: dict[str, list[Union[str, int]]] = {
            "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"], ## VGG16 Model 
            "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
        }
                
        # Features, Avg Pooling, and Classifier
        self.features = self.make_layers(cfgs["D"], self.layer, batch_norm=True)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, self.num_classes),
        )
        
        
        self.to(self.device)
        self.name = f"{self.model}_{self.layer}"
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def summary(self): 
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            summary(self, input_size=self.img_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            self.to(original_device)
    
    def make_layers(self, cfg: list[Union[str, int]], 
                    conv_layer: nn.Module = nn.Conv2d,
                    batch_norm: bool = False) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_size = 224
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                current_size //= 2
            else:
                v = cast(int, v)
                
                # Conv2d 
                if conv_layer == "Conv2d": 
                    conv = nn.Conv2d(self.in_channels, v, kernel_size=3, padding=1)
                    
                # Conv2d_NN (all, random)
                if conv_layer == "ConvNN" and self.sampling != "Spatial":
                    if self.samples == "all":
                        conv = Conv2d_NN(self.in_channels, v, K=9, stride=9, padding=0, 
                                        shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples=self.samples,
                                        magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                    else: 
                        conv = Conv2d_NN(self.in_channels, v, K=9, stride=9, padding=0, 
                                        shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples=int((current_size/4)**2),
                                        magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                    
                # Conv2d_NN (spatial)
                if conv_layer == "ConvNN" and self.sampling == "Spatial":
                    conv = Conv2d_NN_Spatial(self.in_channels, v, K=9, stride=9, padding=0, 
                                            shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples=int(current_size/4),
                                            sample_padding=0,
                                            magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                
                # Conv2d_NN_Attn (all, random) 
                if conv_layer == "ConvNN_Attn" and self.sampling != "Spatial":
                    if self.samples == "all":
                        conv = Conv2d_NN_Attn(self.in_channels, v, K=9, stride=9, padding=0, 
                                            shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples="all",
                                            image_size=(current_size, current_size),
                                            magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                    else: 
                        conv = Conv2d_NN_Attn(self.in_channels, v, K=9, stride=9, padding=0, 
                        shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples=int((current_size/4)**2),
                        image_size=(current_size, current_size),
                        magnitude_type=self.magnitude_type, location_channels=self.location_channels)              
                if conv_layer == "ConvNN_Attn" and self.sampling == "Spatial":
                    conv = Conv2d_NN_Attn_Spatial(self.in_channels, v, K=9, stride=9, padding=0,  
                                                shuffle_pattern=self.shuffle_pattern,shuffle_scale=self.huffle_scale, samples=int(current_size/4),
                                                samples_padding=0,
                                                image_size=(current_size, current_size),
                                                magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                    
                # Attention
                if conv_layer == "Attention":
                    conv = Attention2d(self.in_channels, 
                                       v, num_heads=4, 
                                    shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale,
                                    location_channels=self.location_channels)
                
                # Branching Conv2d + Conv2d_NN (all, random) 
                if conv_layer == "Conv2d/ConvNN" and self.sampling != "Spatial":
                    if self.samples == "all":
                        conv = Conv2d_ConvNN_Branching(self.in_channels, 
                                                    v, 
                                                    channel_ratio=(v, v),
                                                    kernel_size=3, 
                                                    K=9, 
                                                    shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples="all",
                                                    magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                    else:
                        conv = Conv2d_ConvNN_Branching(self.in_channels, 
                                                    v, 
                                                    channel_ratio=(v, v),
                                                    kernel_size=3, 
                                                    K=9, 
                                                    shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples=int((current_size/4)**2),
                                                    magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                if conv_layer == "Conv2d/ConvNN" and self.sampling == "Spatial":
                    conv = Conv2d_ConvNN_Spatial_Branching(self.in_channels, 
                                                        v, 
                                                        channel_ratio=(v, v),
                                                        kernel_size=3, 
                                                        K=9, 
                                                        shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples=int(current_size/4),
                                                        magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                # Branching Conv2d + Conv2d_NN_Attn (all, random)
                if conv_layer == "Conv2d/ConvNN_Attn" and self.sampling != "Spatial":
                    if self.samples == "all":
                        conv = Conv2d_ConvNN_Attn_Branching(self.in_channels, 
                                                        v, 
                                                        channel_ratio=(v, v),
                                                        kernel_size=3, 
                                                        K=9, 
                                                        shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples="all",
                                                        image_size=(current_size, current_size),
                                                        magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                    else:
                        conv = Conv2d_ConvNN_Attn_Branching(self.in_channels, 
                                                        v, 
                                                        channel_ratio=(v, v),
                                                        kernel_size=3, 
                                                        K=9, 
                                                        shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples=int((current_size/4)**2),
                                                        image_size=(current_size, current_size),
                                                        magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                
                # Branching Conv2d + Conv2d_NN_Attn (spatial)
                if conv_layer == "Conv2d/ConvNN_Attn" and self.sampling == "Spatial":
                    conv = Conv2d_ConvNN_Attn_Spatial_Branching(self.in_channels, 
                                                            v, 
                                                            channel_ratio=(v, v),
                                                            kernel_size=3, 
                                                            K=9, 
                                                            shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples=int(current_size/4),
                                                            image_size=(current_size, current_size),
                                                            magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                
                # Branching Attention + Conv2d_NN (all, random)
                if conv_layer == "Attention/ConvNN" and self.sampling != "Spatial":
                    if self.samples == "all":
                        conv = Attention_ConvNN_Branching(self.in_channels, 
                                                        v, 
                                                        channel_ratio=(v, v),
                                                        num_heads=4, 
                                                        K=9, 
                                                        shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples="all",
                                                        magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                    else: 
                        conv = Attention_ConvNN_Branching(self.in_channels, 
                                                        v, 
                                                        channel_ratio=(v, v),
                                                        num_heads=4, 
                                                        K=9, 
                                                        shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples=int((current_size/4)**2),
                                                        magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                    
                # Branching Attention + Conv2d_NN (spatial)
                if conv_layer == "Attention/ConvNN" and self.sampling == "Spatial":
                    conv = Attention_ConvNN_Spatial_Branching(self.in_channels, 
                                                            v, 
                                                            channel_ratio=(v, v),
                                                            num_heads=4, 
                                                            K=9, 
                                                            shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples=int(current_size/4),
                                                            magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                
                # Branching Attention + Conv2d_NN_Attn (all, random)
                
                if conv_layer == "Attention/ConvNN_Attn" and self.sampling != "Spatial":
                    if self.samples == "all":
                        conv = Attention_ConvNN_Attn_Branching(self.in_channels, 
                                                            v, 
                                                            channel_ratio=(v, v),
                                                            num_heads=4, 
                                                            K=9, 
                                                            shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples="all",
                                                            image_size=(current_size, current_size),
                                                            magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                    else: 
                        conv = Attention_ConvNN_Attn_Branching(self.in_channels, 
                                                            v, 
                                                            channel_ratio=(v, v),
                                                            num_heads=4, 
                                                            K=9, 
                                                            shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples=int((current_size/4)**2),
                                                            image_size=(current_size, current_size),
                                                            magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                if conv_layer == "Attention/ConvNN_Attn" and self.sampling == "Spatial":
                    conv = Attention_ConvNN_Attn_Spatial_Branching(self.in_channels, 
                                                                v, 
                                                                channel_ratio=(v, v),
                                                                num_heads=4, 
                                                                K=9, 
                                                                shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale, samples=int(current_size/4),
                                                                image_size=(current_size, current_size),
                                                                magnitude_type=self.magnitude_type, location_channels=self.location_channels)
                # Branching Attention + Conv2d 
                if conv_layer == "Conv2d/Attention":
                    conv = Attention_Conv2d_Branching(self.in_channels, 
                                                    v, 
                                                    channel_ratio=(v, v),
                                                    num_heads=4, 
                                                    kernel_size=3, 
                                                    shuffle_pattern=self.shuffle_pattern, shuffle_scale=self.shuffle_scale,
                                                    location_channels=self.location_channels)
                
                
                if batch_norm:
                    layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
                self.in_channels = v
        return nn.Sequential(*layers)        
                
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
        

    
if __name__ == "__main__":
    import torch
    from types import SimpleNamespace
    
    # ViT-Small configuration
    args = SimpleNamespace(
        img_size = (3, 32, 32),       # (channels, height, width)
        num_classes = 100,              # CIFAR-100 classes
        dropout = 0.1,                  # Dropout rate
        magnitude_type = "similarity",  # Or "distance"
        shuffle_pattern = "NA",         # Default pattern
        shuffle_scale = 1,              # Default scale
        layer = "ConvNN",            # Attention or ConvNN
        sampling = "All",
        location_channels = False,   # Use location channels 
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else "cpu"),
        model = "VGG"                   # Model type
    )
    
    # Create the model
    model = VGG(args)
    
    print("Regular Attention")
    # Print parameter count
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    x = torch.randn(64, 3, 32, 32).to(args.device)  
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}\n")
    
    
    print("ConvNN")
    args.layer = "ConvNN"
    model_convnn = VGG(args)
    total_params, trainable_params = model_convnn.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    output_convnn = model_convnn(x)
    
    print(f"Output shape: {output_convnn.shape}")
