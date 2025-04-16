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
    def __init__(self, in_ch=3, num_classes=100, device="mps"):
        super(CNN, self).__init__()
        # Increased width and added a layer
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_ch, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8), # Added BatchNorm
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), # Added BatchNorm
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(32), # Added BatchNorm
            nn.ReLU(inplace=True),

            # Block 4 (New)
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(48), # Added BatchNorm
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()
        
        # Calculate the flattened size: 512 channels * 2 * 2 spatial dimensions
        flattened_size = 48 * 32 * 32
        
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(flattened_size, num_classes), # Increased intermediate size
        )

        self.device = device
        self.to(self.device)
        self.name = "CNN_DeeperWider" # Renamed for clarity


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
def cnn(pretrained=False, **kwargs):
    """CNN model with comparable complexity to deit_tiny for CIFAR-100"""
    # Ensure this function now uses the updated CNN class
    model = CNN(num_classes=100) # Pass num_classes if needed, device handled internally
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
    cnn_model = cnn()
    
    # Compare parameter counts
    cnn_params = count_parameters(cnn_model)
    
    print(f"CNN parameters: {cnn_params:,}")
    
    # Test with dummy input
    x = torch.randn(1, 3, 32, 32).to('mps')
    cnn_output = cnn_model(x)
    
    print(f"CNN output shape: {cnn_output.shape}")
