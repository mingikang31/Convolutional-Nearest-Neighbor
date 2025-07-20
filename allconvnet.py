import torch 
import torch.nn as nn
from torchsummary import summary


from layers2d import (
    Conv2d_NN, 
    Conv2d_NN_Attn, 
    Attention2d, 
    Conv2d_ConvNN_Branching, 
    Conv2d_ConvNN_Attn_Branching,
    Attention_ConvNN_Branching, 
    Attention_ConvNN_Attn_Branching,
    Attention_Conv2d_Branching
)


class AllConvNet(nn.Module): 
    def __init__(self, args): 
        super(AllConvNet, self).__init__()
        self.args = args
        self.model = "All Convolutional Network"
        self.name = f"{self.model} {self.args.layer}"
        
        layers = []
        in_ch = self.args.img_size[0] 

        for i in range(self.args.num_layers):
            out_ch = self.args.channels[i]

            # A dictionary to hold parameters for the current layer
            layer_params = {
                "in_channels": in_ch,
                "out_channels": out_ch,
                "shuffle_pattern": self.args.shuffle_pattern,
                "shuffle_scale": self.args.shuffle_scale,
            }

            if self.args.layer == "Conv2d":
                layer = nn.Conv2d(
                    in_channels=in_ch, 
                    out_channels=out_ch, 
                    kernel_size=self.args.kernel_size, 
                    stride=1, 
                    padding='same'
                )
            
            elif self.args.layer == "ConvNN":
                layer_params.update({
                    "K": self.args.K,
                    "stride": self.args.K, # Stride is always K
                    "sampling_type": self.args.sampling_type,
                    "num_samples": self.args.num_samples,
                    "sample_padding": self.args.sample_padding,
                    "magnitude_type": self.args.magnitude_type,
                    "coordinate_encoding": self.args.coordinate_encoding
                })
                layer = Conv2d_NN(**layer_params)

            elif self.args.layer == "ConvNN_Attn":
                layer_params.update({
                    "K": self.args.K,
                    "stride": self.args.K,
                    "sampling_type": self.args.sampling_type,
                    "num_samples": self.args.num_samples,
                    "sample_padding": self.args.sample_padding,
                    "magnitude_type": self.args.magnitude_type,
                    "img_size": self.args.img_size[1:], # Pass H, W
                    "attention_dropout": self.args.attention_dropout,
                    "coordinate_encoding": self.args.coordinate_encoding
                })
                layer = Conv2d_NN_Attn(**layer_params)
            
            elif self.args.layer == "Attention":
                layer_params.update({
                    "num_heads": self.args.num_heads,
                })
                layer = Attention2d(**layer_params)
            elif "/" in self.args.layer: # Handle all branching cases
                ch1 = out_ch // 2 if out_ch % 2 == 0 else out_ch // 2 + 1
                ch2 = out_ch - ch1
                
                layer_params.update({"channel_ratio": (ch1, ch2)})
                
                # --- Check all sub-cases for branching layers ---
                if self.args.layer == "Conv2d/ConvNN":
                    layer_params.update({
                        "kernel_size": self.args.kernel_size,
                        "K": self.args.K, "stride": self.args.K,
                        "sampling_type": self.args.sampling_type, "num_samples": self.args.num_samples,
                        "sample_padding": self.args.sample_padding, "magnitude_type": self.args.magnitude_type,
                        "coordinate_encoding": self.args.coordinate_encoding
                    })
                    layer = Conv2d_ConvNN_Branching(**layer_params)
                
                elif self.args.layer == "Conv2d/ConvNN_Attn":
                    layer_params.update({
                        "kernel_size": self.args.kernel_size,
                        "K": self.args.K, "stride": self.args.K,
                        "sampling_type": self.args.sampling_type, "num_samples": self.args.num_samples,
                        "sample_padding": self.args.sample_padding, "magnitude_type": self.args.magnitude_type,
                        "img_size": self.args.img_size[1:],
                        "coordinate_encoding": self.args.coordinate_encoding
                    })
                    layer = Conv2d_ConvNN_Attn_Branching(**layer_params)
                
                elif self.args.layer == "Attention/ConvNN":
                    layer_params.update({
                        "num_heads": self.args.num_heads,
                        "K": self.args.K, "stride": self.args.K,
                        "sampling_type": self.args.sampling_type, "num_samples": self.args.num_samples,
                        "sample_padding": self.args.sample_padding, "magnitude_type": self.args.magnitude_type,
                        "coordinate_encoding": self.args.coordinate_encoding
                    })
                    layer = Attention_ConvNN_Branching(**layer_params)

                elif self.args.layer == "Attention/ConvNN_Attn":
                    layer_params.update({
                        "num_heads": self.args.num_heads,
                        "K": self.args.K, "stride": self.args.K,
                        "sampling_type": self.args.sampling_type, "num_samples": self.args.num_samples,
                        "sample_padding": self.args.sample_padding, "magnitude_type": self.args.magnitude_type,
                        "img_size": self.args.img_size[1:],
                        "coordinate_encoding": self.args.coordinate_encoding
                    })
                    layer = Attention_ConvNN_Attn_Branching(**layer_params)
                
                # This is the specific case that was failing
                elif self.args.layer == "Conv2d/Attention":
                    layer_params.update({
                        "num_heads": self.args.num_heads,
                        "kernel_size": self.args.kernel_size, 
                        "coordinate_encoding": self.args.coordinate_encoding
                    })
                    layer = Attention_Conv2d_Branching(**layer_params)
                
                else:
                    # This else now only catches unknown branching types
                    raise ValueError(f"Unknown branching layer type: {self.args.layer}")

            else:
                # This is the final else for non-branching types
                raise ValueError(f"Layer type {self.args.layer} not supported in AllConvNet")

            layers.append(nn.InstanceNorm2d(num_features=out_ch)) # Pre-layer normalization
            layers.append(layer)
            if self.args.layer == "ConvNN_Attn":
                pass #layers.append(nn.Dropout(p=self.args.attention_dropout))
            layers.append(nn.ReLU(inplace=True))
            
            # Update in_ch for the next layer
            in_ch = out_ch
            
        self.features = nn.Sequential(*layers)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
            
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_ch, self.args.num_classes) # Use the final in_ch value
        )
        
        self.to(self.args.device)

    def forward(self, x): 
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def summary(self): 
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but img_size doesn't include it
            summary(self, input_size=self.img_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)
        
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch
    from types import SimpleNamespace

    # Assume AllConvNet is in the same file or imported correctly
    # from allconvnet import AllConvNet 

    def run_test(args, x):
        """Helper function to instantiate, test, and print results for a given configuration."""
        header = f"--- Testing Layer: {args.layer} | Sampling: {args.sampling_type.upper()} ---"
        print(header)
        
        try:
            # Create the model with the given arguments
            model = AllConvNet(args)
            
            # Print parameter count
            total_params, trainable_params = model.parameter_count()
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
            # Forward pass
            with torch.no_grad():
                output = model(x)
            
            print(f"Output shape: {output.shape}")
        
        except Exception as e:
            print(f"!!! TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("-" * len(header), "\n")


    # 1. --- Define Base Configuration and Dummy Data ---
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running tests on device: {device}\n")

    base_args = SimpleNamespace(
        num_layers=3,
        channels=[16, 32, 64],
        img_size=(3, 32, 32),
        num_classes=10,
        
        # Conv/NN Layer Params
        kernel_size=3,
        K=9,
        magnitude_type="similarity",
        sample_padding=0,
        
        # Attention Params
        num_heads=4,
        
        # Shuffle Params
        shuffle_pattern="BA",
        shuffle_scale=2,
        
        # Branching Params
        channel_ratio=(1, 1), # 50/50 split
        
        # Device
        device=device, 
        coordinate_encoding=True,  # Default to False for simplicity
    )

    # Dummy input tensor
    x = torch.randn(4, 3, 32, 32).to(device)

    base_args.layer = "Conv2d"  # Start with Conv2d layer
    base_args.sampling_type = "N/A"  # Not applicable for Conv2d
    base_args.num_samples = -1  # Not applicable for Conv2d
    run_test(base_args, x)
    

    # 2. --- Run Tests for All Layer Variants ---

    # # Test (1): Conv2d_NN
    # base_args.layer = "ConvNN"
    # # Test with ALL sampling
    # base_args.sampling_type = "all"
    # base_args.num_samples = -1
    # run_test(base_args, x)
    # # Test with RANDOM sampling
    # base_args.sampling_type = "random"
    # base_args.num_samples = 64
    # run_test(base_args, x)
    # # Test with SPATIAL sampling
    # base_args.sampling_type = "spatial"
    # base_args.num_samples = 8
    # run_test(base_args, x)

    # # Test (2): Conv2d_NN_Attn
    # base_args.layer = "ConvNN_Attn"
    # # Test with ALL sampling
    # base_args.sampling_type = "all"
    # base_args.num_samples = -1
    # run_test(base_args, x)
    # # Test with RANDOM sampling
    # base_args.sampling_type = "random"
    # base_args.num_samples = 64
    # run_test(base_args, x)
    # # Test with SPATIAL sampling
    # base_args.sampling_type = "spatial"
    # base_args.num_samples = 8
    # run_test(base_args, x)

    # # Test (3): Attention2d (no sampling variants)
    # base_args.layer = "Attention"
    # base_args.sampling_type = "N/A" # Not applicable
    # base_args.num_samples = -1
    # run_test(base_args, x)

    # # Test (4): Conv2d_ConvNN_Branching
    # base_args.layer = "Conv2d/ConvNN_Branching"
    # # Test with ALL sampling
    # base_args.sampling_type = "all"
    # base_args.num_samples = -1
    # run_test(base_args, x)
    # # Test with RANDOM sampling
    # base_args.sampling_type = "random"
    # base_args.num_samples = 64
    # run_test(base_args, x)
    # # Test with SPATIAL sampling
    # base_args.sampling_type = "spatial"
    # base_args.num_samples = 8
    # run_test(base_args, x)

    # # Test (5): Conv2d_ConvNN_Attn_Branching
    # base_args.layer = "Conv2d/ConvNN_Attn_Branching"
    # # Test with ALL sampling
    # base_args.sampling_type = "all"
    # base_args.num_samples = -1
    # run_test(base_args, x)
    # # Test with RANDOM sampling
    # base_args.sampling_type = "random"
    # base_args.num_samples = 64
    # run_test(base_args, x)
    # # Test with SPATIAL sampling
    # base_args.sampling_type = "spatial"
    # base_args.num_samples = 8
    # run_test(base_args, x)

    # # Test (6): Attention_ConvNN_Branching
    # base_args.layer = "Attention/ConvNN_Branching"
    # # Test with ALL sampling
    # base_args.sampling_type = "all"
    # base_args.num_samples = -1
    # run_test(base_args, x)
    # # Test with RANDOM sampling
    # base_args.sampling_type = "random"
    # base_args.num_samples = 64
    # run_test(base_args, x)
    # # Test with SPATIAL sampling
    # base_args.sampling_type = "spatial"
    # base_args.num_samples = 8
    # run_test(base_args, x)

    # # Test (7): Attention_ConvNN_Attn_Branching
    # base_args.layer = "Attention/ConvNN_Attn_Branching"
    # # Test with ALL sampling
    # base_args.sampling_type = "all"
    # base_args.num_samples = -1
    # run_test(base_args, x)
    # # Test with RANDOM sampling
    # base_args.sampling_type = "random"
    # base_args.num_samples = 64
    # run_test(base_args, x)
    # # Test with SPATIAL sampling
    # base_args.sampling_type = "spatial"
    # base_args.num_samples = 8
    # run_test(base_args, x)

    # # Test (8): Attention_Conv2d_Branching (no sampling variants)
    # base_args.layer = "Conv2d/Attention"
    # base_args.num_samples = -1
    # run_test(base_args, x)
