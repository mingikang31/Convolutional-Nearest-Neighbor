# Convolutional Nearest Neighbor (ConvNN) with AllConvNet and Vision Transformer Architectures 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Grants & Funding
- **Christenfeld Summer Research Fellowship**, Bowdoin College
- **Google AI 2024 Funding**, Last Mile Fund
- **NYC Stem Funding**, Last Mile Fund

**Project periods:** Summer 2024, Spring 2025, Summer 2025

## Overview 
Traditional convolutions only see a fixed grid of adjacent pixels. **ConvNN**: 
- Finds the _K_-nearest neighbors of each token (pixel or feature vector). 
- Aggregates their values by distance- or similarity-based weightings. 
- Captures long-range dependencies and relationships between pixels.
- Is drop-in compatible with both AllConvNet and Vision Transformer architectures.

### Key Concepts
- **AllConvNet Architecture**: A fully convolutional network that can incorporate various layer types including traditional Conv2d, ConvNN, and Attention layers.
- **Vision Transformer (ViT)**: Transformer-based architecture with patch embeddings that can utilize ConvNN layers for enhanced feature extraction.
- **Layer Flexibility**: Supports single layer types or branching layer types (e.g., Conv2d/ConvNN, ConvNNAttention).
- **Attention Integration**: Can combine attention mechanisms with convolutional nearest neighbor layers for enhanced feature extraction.

### Implementation 
Two main architectures are supported:

**AllConvNet Layers:**
- **Conv2d**: Traditional 2D convolution layers
- **ConvNN**: Nearest neighbor convolution layers
- **ConvNN_Attn**: ConvNN with attention mechanisms
- **Attention**: Pure attention layers
- **Branching Combination**: Branching layer types (e.g., Conv2d/ConvNN, Attention/ConvNN)

**ViT Layers:**
- **MultiHeadAttention**: Standard transformer multi-head attention
- **MultiHeadConvNNAttention**: 1D nearest neighbor convolution with linear projection on feature vectors
- **MultiHeadConvNN**: 1D nearest Neighbor convolution with linear projects on sequences
- **MultiHeadConv1dAttention**: Traditional 1D convolution with linear projection on feature vectors
- **MultiHeadConv1d**: Traditional 1D convolution with linear projection on sequences

## Installation
```Shell 
git clone https://github.com/mingikang31/Convolutional-Nearest-Neighbor.git
```
Then, Install required dependencies:
- torch, torchvision, torchsummary, numpy, matplotlib, tqdm, Pillow
```Shell
pip install -r requirements.txt
```

## Training & Evaluation 

### AllConvNet Examples

Basic ConvNN training on CIFAR-10:
```Shell 
python allconvnet_main.py --layer ConvNN --num_layers 3 --channels 8 16 32 --K 9 --sampling Spatial --num_samples 64 --batch_size 64 --output_dir ./Output/AllConvNet/ConvNN --device cuda --num_epochs 10 --dataset cifar10
```

Hybrid Conv2d/ConvNN layers:
```Shell
python allconvnet_main.py --layer Conv2d/ConvNN --num_layers 3 --channels 8 16 32 --sampling Spatial --num_samples 8 --dataset cifar10 --num_epochs 10 --device cuda --output_dir ./Output/AllConvNet/Conv2d_ConvNN_Spatial
```

### Vision Transformer Examples

Standard ViT with attention:
```Shell
python vit_main.py --layer Attention --patch_size 16 --num_layers 3 --num_heads 4 --d_model 8 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --output_dir ./Output/VIT/VIT_Attention
```

ViT with ConvNN layers:
```Shell
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 3 --num_heads 4 --d_model 8 --dropout 0.1 --K 9 --num_samples 32 --dataset cifar10 --num_epochs 10 --output_dir ./Output/VIT/VIT_ConvNN
```

ViT with ConvNN and Attention combined:
```Shell
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 3 --num_heads 4 --d_model 8 --dropout 0.1 --attention_dropout 0.1 --K 9 --num_samples 32 --dataset cifar10 --num_epochs 10 --output_dir ./Output/VIT/VIT_ConvNNAttention
```

## Command-Line Interface 

### AllConvNet (`allconvnet_main.py`)
Run `python allconvnet_main.py --help` to see all available options.

#### Model & Layer Configuration
| Flag                 | Default       | Choices                   | Description                                  |
| -------------------- | ------------- | ------------------------- | -------------------------------------------- |
| `--layer`            | `ConvNN`      | `Conv2d`, `ConvNN`, `ConvNN_Attn`, `Attention`, `Conv2d/ConvNN`, `Conv2d/ConvNN_Attn`, `Attention/ConvNN`, `Attention/ConvNN_Attn`, `Conv2d/Attention` | Which convolution/attention layer to use |
| `--num_layers`       | `5`           | _integer_                 | Number of sequential layers                  |
| `--channels`         | `[32,64,128,256,512]` | _list of integers_    | Channel sizes for each layer                 |
| `--kernel_size`      | `3`           | _integer_                 | Kernel size for Conv2d layers               |

### Vision Transformer (`vit_main.py`)
Run `python vit_main.py --help` to see all available options.

#### Model & Layer Configuration
| Flag                 | Default       | Choices                   | Description                                  |
| -------------------- | ------------- | ------------------------- | -------------------------------------------- |
| `--layer`            | `Attention`   | `Attention`, `ConvNN`, `ConvNNAttention`, `Conv1d`, `Conv1dAttention` | Layer type for ViT transformer blocks |
| `--patch_size`       | `16`          | _integer_                 | Patch size for input images                 |
| `--num_layers`       | `8`           | _integer_                 | Number of transformer layers                |
| `--num_heads`        | `4`           | _integer_                 | Number of attention heads                   |
| `--d_model`          | `512`         | _integer_                 | Model dimension                             |
| `--dropout`          | `0.1`         | _float_                   | General dropout rate                        |
| `--attention_dropout`| `0.1`         | _float_                   | Attention-specific dropout rate             |

### ConvNN-Specific Parameters (Both Architectures)
| Flag                 | Default       | Choices                   | Description                                  |
| -------------------- | ------------- | ------------------------- | -------------------------------------------- |
| `--K`                | `9`           | _integer_                 | Number of nearest neighbors                  |
| `--sampling`         | `None`/`All`  | `All`, `Random`, `Spatial`| How to select neighbors (AllConvNet only)   |
| `--num_samples`      | `0`           | _integer_                 | Max samples per query (0 means all)         |
| `--kernel_size`      | `3`           | _integer_                 | Kernel size for Conv layers                 |
| `--magnitude_type`   | `similarity`  | `similarity`, `distance`  | Weighting type                               |
| `--shuffle_pattern`  | `BA`          | `BA`, `NA`                | Before-After vs no shuffle (AllConvNet only)|
| `--shuffle_scale`    | `2`           | _integer_                 | Pixel unshuffle factor (AllConvNet only)    |
| `--location_channels`| _flag_        |                           | Append XY coordinates (AllConvNet only)     |

### Data & Training (Both Architectures)
| Flag                 | Default        | Choices                   | Description                                  |
| -------------------- | -------------- | ------------------------- | -------------------------------------------- |
| `--dataset`          | `cifar10`      | `cifar10`, `cifar100`, `imagenet`| Dataset for training and evaluation   |
| `--data_path`        | `./Data`       | _string_                  | Path to the dataset                          |
| `--batch_size`       | `64`           | _integer_                 | Batch size for training                      |
| `--num_epochs`       | `100`          | _integer_                 | Number of epochs for training                |
| `--use_amp`          | _flag_         |                           | Enable mixed-precision (FP16)                |
| `--clip_grad_norm`   | `None`         | _float_                   | Maximum gradient norm (if any)               |

### Optimization & Learning Rate (Both Architectures)
| Flag                 | Default        | Choices                     | Description                                  |
| -------------------- | -------------- | --------------------------- | -------------------------------------------- |
| `--criterion`        | `CrossEntropy` | `CrossEntropy`, `MSE`       | Loss function                                |
| `--optimizer`        | `adamw`        | `adam`, `sgd`, `adamw`      | Optimizer                                    |
| `--momentum`         | `0.9`          | _float_                     | Only for SGD                                 |  
| `--weight_decay`     | `1e-6`         | _float_                     | L2 regularization                            |
| `--lr`               | `1e-3`         | _float_                     | Base learning rate                           |
| `--scheduler`        | `step`         | `step`, `cosine`, `plateau` | LR schedule type                             |
| `--lr_step`          | `20`           | _integer_                   | Step scheduler step size                     |
| `--lr_gamma`         | `0.1`          | _float_                     | Step scheduler decay factor                  |

### Device & Reproduction (Both Architectures)
| Flag                 | Default        | Choices                     | Description                                  |
| -------------------- | -------------- | --------------------------- | -------------------------------------------- |
| `--device`           | `cuda`         | `cuda`, `mps`, `cpu`        | Device for training inference                |
| `--seed`             | `0`            | _integer_                   | Random seed for reproducibility              |
| `--output_dir`       | Architecture-specific | _path_               | Path to save output                          |  

## Project Structure 
```Shell
.
├── layers1d.py           # 1D layers for ViT and AllConvNet
├── layers2d.py           # 2D layers for AllConvNet
├── dataset.py            # CIFAR-10/100 & ImageNet wrappers
├── train_eval.py         # Training & evaluation loop
├── allconvnet.py         # AllConvNet architecture implementation
├── allconvnet_main.py    # CLI entrypoint for AllConvNet
├── vit.py                # Vision Transformer implementation
├── vit_main.py           # CLI entrypoint for ViT
├── utils.py              # I/O, logging, seed setup
├── README.md             # ← you are here
└── LICENSE               # MIT License
```

## Architecture Comparison

| Feature | AllConvNet | Vision Transformer |
|---------|------------|-------------------|
| **Input Processing** | Direct pixel-level convolution | Patch-based tokenization |
| **Layer Types** | Conv2d, ConvNN, Attention, Hybrids | Attention, ConvNN, Conv1d, Hybrids |
| **Spatial Processing** | 2D spatial operations | 1D sequence of patches |
| **Configuration** | Channel-based depth control | Transformer block depth |
| **Best For** | Image-level feature extraction | Global context modeling |

## Example Workflows

### Comparing Architectures
```Shell
# AllConvNet with ConvNN
python allconvnet_main.py --layer ConvNN --num_layers 3 --channels 32 64 128 --dataset cifar10 --output_dir ./Output/AllConvNet/ConvNN

# ViT with ConvNN
python vit_main.py --layer ConvNN --num_layers 3 --d_model 128 --dataset cifar10 --output_dir ./Output/ViT/ConvNN
```

### Ablation Studies
```Shell
# Pure attention (ViT)
python vit_main.py --layer Attention --num_layers 6 --num_heads 8 --d_model 256

# ConvNN + Attention (ViT)
python vit_main.py --layer ConvNNAttention --num_layers 6 --num_heads 8 --d_model 256 --K 16

# Hybrid Conv2d/ConvNN (AllConvNet)
python allconvnet_main.py --layer Conv2d/ConvNN --channels 64 128 256 --sampling Random --num_samples 32
```

## License 
Convolutional-Nearest-Neighbor is released under the MIT License. Please see the [LICENSE](LICENSE) file for more information.

## Contributing 
Contributions, issues, and feature requests are welcome! 
Please reach out to: 
- **Mingi Kang** <mkang2@bowdoin.edu>
- **Jeova Farias** <j.farias@bowdoin.edu>