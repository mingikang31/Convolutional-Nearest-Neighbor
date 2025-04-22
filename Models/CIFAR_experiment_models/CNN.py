import torch
import torch.nn as nn
from functools import partial
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch
import torch.nn as nn
from functools import partial
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, VisionTransformer # Import VisionTransformer here
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16, num_layers=2, num_classes=100, device="mps"):
        super(CNN, self).__init__()
        # Increased width and added a layer
        
        assert num_layers >= 2, "Number of layers must be at least 2"
        assert mid_ch >= 8, "Middle channels must be at least 8"
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())
            else: 
                layers.append(nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1))
                # layers.append(nn.BatchNorm2d(mid_ch))
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
        self.name = "CNN" # Renamed for clarity


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        # Ensure the model is on CPU for torchsummary if it causes issues with MPS
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
def cnn_100(pretrained=False, **kwargs):
    """CNN model with comparable complexity to deit_tiny for CIFAR-100"""
    # Ensure this function now uses the updated CNN class
    model = CNN(in_ch=3, mid_ch=16, num_layers=2, num_classes=100, device='mps')
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def cnn_10(pretrained=False, **kwargs):
    """CNN model with comparable complexity to deit_tiny for CIFAR-10"""
    # Ensure this function now uses the updated CNN class
    model = CNN(in_ch=3, mid_ch=16, num_layers=2, num_classes=10, device='mps')
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
        cnn_100(),
        cnn_10(),
        
    ]
    
    x = torch.randn(1, 3, 32, 32).to('mps')

    for model in models:
        print(f"Model: {model.name}")
        print(f"Parameters: {count_parameters(model):,}")
        print(f"Output shape: {model(x).shape}")
        print("-" * 30)
    