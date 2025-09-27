import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NonLocalBlock(nn.Module):
    """
    Non-Local Block from "Non-local Neural Networks" (Wang et al., CVPR 2018)
    
    This is a plug-and-play block that can be inserted into any CNN architecture
    to capture long-range dependencies.
    """
    
    def __init__(self, in_channels, inter_channels=None, dimension=3, 
                 sub_sample=True, bn_layer=True, mode='embedded_gaussian'):
        """
        Initialize Non-Local Block.
        
        Args:
            in_channels: Number of input channels
            inter_channels: Number of channels in intermediate layers (default: in_channels//2)
            dimension: 1D, 2D or 3D for different data types
            sub_sample: Whether to use pooling to reduce computation
            bn_layer: Whether to use BatchNorm
            mode: Type of non-local operation ('gaussian', 'embedded_gaussian', 'dot_product', 'concatenation')
        """
        super(NonLocalBlock, self).__init__()
        
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.mode = mode
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # Create convolution layers based on dimension
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:  # dimension == 1
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            bn = nn.BatchNorm1d
        
        # g transformation
        self.g = conv_nd(in_channels=self.in_channels, 
                        out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
        
        # W_z transformation (output)
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                       out_channels=self.in_channels,
                       kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            # Initialize BN to zero for identity mapping
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                           out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        
        # θ and φ transformations (for embedded versions)
        if self.mode in ['embedded_gaussian', 'dot_product', 'concatenation']:
            self.theta = conv_nd(in_channels=self.in_channels,
                               out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        
        # Concatenation-specific layers
        if mode == 'concatenation':
            self.concat_project = nn.Sequential(
                nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
                nn.ReLU()
            )
        
        # Sub-sampling (optional, for efficiency)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode in ['embedded_gaussian', 'dot_product', 'concatenation']:
                self.phi = nn.Sequential(self.phi, max_pool_layer)
    
    def forward(self, x):
        """
        Forward pass of Non-Local Block.
        
        Args:
            x: Input tensor (B, C, T, H, W) for 3D or (B, C, H, W) for 2D
        
        Returns:
            Output tensor with residual connection
        """
        batch_size = x.size(0)
        
        # g transformation - value
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # (B, HW, C)
        
        if self.mode == 'gaussian':
            # Standard Gaussian
            theta_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)  # (B, HW, C)
            phi_x = x.view(batch_size, self.in_channels, -1)  # (B, C, HW)
            f = torch.matmul(theta_x, phi_x)  # (B, HW, HW)
            f_div_C = F.softmax(f, dim=-1)
            
        elif self.mode == 'embedded_gaussian':
            # Embedded Gaussian (used in the paper)
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)  # (B, HW, C)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # (B, C, HW)
            f = torch.matmul(theta_x, phi_x)  # (B, HW, HW)
            f_div_C = F.softmax(f, dim=-1)
            
        elif self.mode == 'dot_product':
            # Dot product (without softmax)
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)  # (B, HW, C)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # (B, C, HW)
            f = torch.matmul(theta_x, phi_x)  # (B, HW, HW)
            N = f.size(-1)
            f_div_C = f / N  # Normalize by number of positions
            
        elif self.mode == 'concatenation':
            # Concatenation (Relation Networks style)
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            # Concatenate all pairs
            theta_x = theta_x.repeat(1, 1, 1, phi_x.size(3))
            phi_x = phi_x.repeat(1, 1, theta_x.size(2), 1)
            
            concat_features = torch.cat([theta_x, phi_x], dim=1)
            f = self.concat_project(concat_features)
            b, _, h, w = f.size()
            f = f.view(b, h, w)
            
            N = f.size(-1)
            f_div_C = f / N
        
        # Apply attention to values
        y = torch.matmul(f_div_C, g_x)  # (B, HW, C)
        y = y.permute(0, 2, 1).contiguous()  # (B, C, HW)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # (B, C, T, H, W) or (B, C, H, W)
        
        # Output transformation
        W_y = self.W(y)
        
        # Residual connection
        z = W_y + x
        
        return z


class NonLocalNet(nn.Module):
    """
    Example network with Non-Local blocks for video classification.
    Based on ResNet + Non-Local blocks architecture from the paper.
    """
    
    def __init__(self, num_classes=400, num_blocks=5):
        super(NonLocalNet, self).__init__()
        
        # Simple backbone (you would use ResNet in practice)
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), 
                               stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), 
                                    stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Add conv blocks
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(512)
        
        # Add Non-Local blocks
        self.nl_blocks = nn.ModuleList()
        for i in range(num_blocks):
            if i < 3:
                # Add to res3
                self.nl_blocks.append(NonLocalBlock(256, dimension=3))
            else:
                # Add to res4
                self.nl_blocks.append(NonLocalBlock(512, dimension=3))
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Input: (B, C, T, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Apply first 3 Non-Local blocks
        for i in range(min(3, len(self.nl_blocks))):
            x = self.nl_blocks[i](x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        # Apply remaining Non-Local blocks
        for i in range(3, len(self.nl_blocks)):
            x = self.nl_blocks[i](x)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x


def visualize_attention(model, video_clip):
    """
    Visualize attention weights from Non-Local blocks.
    """
    import matplotlib.pyplot as plt
    
    # Hook to capture attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        if hasattr(module, 'attention'):
            attention_weights.append(module.attention.detach().cpu())
    
    # Register hooks
    for nl_block in model.nl_blocks:
        nl_block.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        output = model(video_clip)
    
    # Visualize attention from first Non-Local block
    if attention_weights:
        att = attention_weights[0][0]  # First sample, first block
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # Show attention for 4 different query positions
        positions = [0, 100, 500, 1000]
        for idx, pos in enumerate(positions):
            ax = axes[idx // 2, idx % 2]
            ax.imshow(att[pos].reshape(32, 32), cmap='hot')
            ax.set_title(f'Attention from position {pos}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return attention_weights


# Example usage
def demo():
    """
    Demonstrate Non-Local blocks in action.
    """
    # Create a simple 2D Non-Local block
    nl_block_2d = NonLocalBlock(in_channels=256, dimension=2)
    
    # Create input tensor (batch_size=2, channels=256, height=14, width=14)
    x = torch.randn(2, 256, 14, 14)
    
    # Apply Non-Local block
    output = nl_block_2d(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Shapes match (residual connection): {x.shape == output.shape}")
    
    # For video (3D)
    nl_block_3d = NonLocalBlock(in_channels=512, dimension=3)
    video_input = torch.randn(2, 512, 4, 7, 7)  # (B, C, T, H, W)
    video_output = nl_block_3d(video_input)
    
    print(f"\nVideo input shape: {video_input.shape}")
    print(f"Video output shape: {video_output.shape}")
    
    # Compare computational cost
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameters in 2D Non-Local block: {count_parameters(nl_block_2d):,}")
    print(f"Parameters in 3D Non-Local block: {count_parameters(nl_block_3d):,}")
    
    # Show different modes
    modes = ['gaussian', 'embedded_gaussian', 'dot_product']
    for mode in modes:
        nl = NonLocalBlock(256, dimension=2, mode=mode)
        out = nl(x)
        print(f"\nMode '{mode}' - Output shape: {out.shape}")


if __name__ == "__main__":
    demo()
    
    # Example: Create a video classification model
    model = NonLocalNet(num_classes=400, num_blocks=5)
    
    # Create dummy video input (2 clips, 3 channels, 32 frames, 112x112 pixels)
    video = torch.randn(2, 3, 32, 112, 112)
    
    # Forward pass
    predictions = model(video)
    print(f"\nVideo classification model:")
    print(f"Input shape: {video.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")