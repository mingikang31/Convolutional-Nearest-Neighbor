# Convolutional Nearest Neighbor (ConvNN) for Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Paper 
- **Attention Via Convolutional Nearest Neighbors**, Mingi Kang, Jeova Farias, [[arXiv](https://arxiv.org/abs/2511.14137)] 
- This repository is complemented by the Convolutional Nearest Neighbor Attention (ConvNN-Attention) repository: [https://github.com/mingikang31/Convolutional-Nearest-Neighbor-Attention](https://github.com/mingikang31/Convolutional-Nearest-Neighbor-Attention)

## Grants & Funding
- **Fall Research Grant**, Bowdoin College
- **Allen B. Tucker Computer Science Research Prize**, Bowdoin College
- **Christenfeld Summer Research Fellowship**, Bowdoin College
- **Google AI 2024 Funding**, Last Mile Fund
- **NYC Stem Funding**, Last Mile Fund

**Project periods:** Summer 2024, Spring 2025, Summer 2025, Fall 2026, Spring 2026

## Overview
Convolutional Nearest Neighbor (ConvNN) is a neural network layer that leverages nearest neighbor search to perform convolution-like operations. Instead of aggregating features from a fixed local neighborhood, ConvNN dynamically selects the most relevant neighboring features based on similarity or distance metrics.

### Key Concepts
- **Convolutional Neural Network Architectures**: Any standard CNN architecture (e.g., ResNet, VGG) can be adapted to use ConvNN layers in place of traditional convolutional layers.
- **Layer Flexibility**: The project supports single-layer types (e.g., `ConvNN` only) and branching architectures (e.g., `Branching`, `Branching_Attn`) to combine the strengths of different operations.
- **Sampling Strategies**: To manage computational complexity, `ConvNN` layers support multiple sampling strategies: `all` (dense), `random`, and `spatial`.

### Implementation

**1. 1D Neural Network Layers (`layers1d.py`)**: These are 1D layers operating directly on image-like feature maps.
- **`Conv1d_NN`**: The core 1D nearest neighbor convolution.
- **`Conv1d_NN_Attn`**: A hybrid that uses Q,K,V projections combined with Conv1d_NN.
- **Branching Combinations**: Parallel branches of any two of the above layers (e.g., `Conv1d/Conv1d_NN`, `Conv1d/Conv1d_NN_Attn`).
- **`PixelShuffle1D`**: 1D Pixel Shuffle layer for upsampling.
- **`PixelUnshuffle1D`**: 1D Pixel Unshuffle layer for downsampling.

**2. 2D Neural Network Layers (`layers2d.py`)**: These are 2D layers operating directly on image-like feature maps.
- **`Conv2d`**: Standard 2D convolution.
- **`Conv2d_NN`**: The core 2D nearest neighbor convolution.
- **`Conv2d_NN_Attn`**: A hybrid that uses Q,K,V projections combined with Conv2d_NN.
- **Branching Combinations**: Parallel branches of any two of the above layers (e.g., `Conv2d/Conv2d_NN`, `Conv2d/Conv2d_NN_Attn`).


## Installation
```shell
git clone [https://github.com/mingikang31/Convolutional-Nearest-Neighbor.git](https://github.com/mingikang31/Convolutional-Nearest-Neighbor.git)
cd Convolutional-Nearest-Neighbor
````

Then, install the required dependencies:

```shell
pip install -r requirements.txt
```
## Command-Line Interface

### Main Script (`main.py`)

Run `python main.py --help` to see all available options.

#### Model & Layer Configuration

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--model` | `vgg11` | `vgg11`, `vgg13`, `vgg16`, `vgg19`, `resnet18`, `resnet34` | Model architecture to use |
| `--layer` | `ConvNN` | `Conv2d`, `Conv2d_New`, `ConvNN`, `ConvNN_Attn`, `Branching`, `Branching_Attn` | Type of convolution or attention layer to use |
| `--kernel_size` | `3` | *integer* | Kernel size for Conv2d layers |
| `--padding` | `1` | *integer* | Padding for convolution layers |

#### ConvNN-Specific Parameters

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--K` | `9` | *integer* | Number of K-nearest neighbors to find |
| `--sampling_type` | `all` | `all`, `random`, `spatial` | How to select the pool of neighbor candidates |
| `--num_samples` | `-1` | *integer* | Number of samples for `random` or `spatial` sampling. Use `-1` for `all` sampling |
| `--sample_padding` | `0` | *integer* | Padding for spatial sampling in ConvNN models |
| `--magnitude_type` | `cosine` | `cosine`, `euclidean` | Distance metric for finding nearest neighbors |
| `--similarity_type` | `Col` | `Loc`, `Col`, `Loc_Col` | Similarity computation type for ConvNN models |
| `--aggregation_type` | `Col` | `Col`, `Loc_Col` | Aggregation type for ConvNN models |
| `--lambda_param` | `0.5` | *float (0-1)* | Lambda parameter for `Loc_Col` aggregation blending |
| `--shuffle_pattern` | `NA` | `BA`, `NA` | Shuffle pattern: `BA` (Before & After) or `NA` (No Shuffle) |
| `--shuffle_scale` | `0` | *integer* | Scale factor for pixel shuffle/unshuffle |

- (note) For `random` sampling type, set `--num_samples` to the desired number of neighbors to sample (e.g., `4`). For `spatial` sampling type, set `--num_samples` to the spatially separated grid (e.g., `3` for 3x3). For dense attention, set `--sampling_type` to `all` and `--num_samples` to `-1`.

#### Branching Layer Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--branch_ratio` | `0.5` | Branch ratio for `Branching` layer (between 0 and 1). Example: `0.25` means 25% of input/output channels go to ConvNN branch, rest to Conv2d branch |
| `--attention_dropout` | `0.1` | Dropout rate for attention layers in `ConvNN_Attn` and `Branching_Attn` |

#### Dataset Configuration

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--dataset` | `cifar10` | `cifar10`, `cifar100`, `imagenet` | Dataset to use for training and evaluation |
| `--data_path` | `./Data` | *path* | Path to the dataset directory |
| `--resize` | `None` | *integer* | Resize images to specified size (e.g., 64 for 64x64) |
| `--augment` | `False` | *flag* | Enable data augmentation |
| `--noise` | `0.0` | *float* | Standard deviation of Gaussian noise to add to data |

#### Training Configuration

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--batch_size` | `256` | *integer* | Batch size for training and evaluation |
| `--num_epochs` | `150` | *integer* | Number of epochs for training |
| `--use_amp` | `False` | *flag* | Enable mixed precision training (automatic mixed precision) |
| `--use_compiled` | `False` | *flag* | Use `torch.compile` for model optimization |
| `--compile_mode` | `default` | `default`, `reduce-overhead`, `reduce-memory`, `max-autotune` | Compilation mode for `torch.compile` |
| `--clip_grad_norm` | `1.0` | *float* | Gradient clipping maximum norm value |

#### Optimization Configuration

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--criterion` | `CrossEntropy` | `CrossEntropy`, `MSE` | Loss function to use for training |
| `--optimizer` | `adamw` | `adam`, `sgd`, `adamw` | Optimizer algorithm |
| `--lr` | `1e-3` | *float* | Initial learning rate |
| `--weight_decay` | `1e-6` | *float* | Weight decay (L2 regularization) for optimizer |
| `--momentum` | `0.9` | *float* | Momentum parameter for SGD optimizer |
| `--scheduler` | `none` | `step`, `cosine`, `plateau`, `none` | Learning rate scheduler type |
| `--lr_step` | `20` | *integer* | Step size for step scheduler (decrease LR every N epochs) |
| `--lr_gamma` | `0.1` | *float* | Multiplicative factor for LR decay in step scheduler |

#### System Configuration

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--device` | `cuda` | `cpu`, `cuda`, `mps` | Device to use for training and evaluation |
| `--seed` | `0` | *integer* | Random seed for reproducibility |

#### Output Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--output_dir` | `./Output/VGG/ConvNN` | Directory to save output files (results, model info, logs) |
| `--test_only` | `False` | *flag* | Only test the model without training |

## Training Examples

### Train VGG11 with Standard Conv2d

```shell
python main.py \
    --model vgg11 \
    --layer Conv2d \
    --kernel_size 3 \
    --dataset cifar10 \
    --optimizer adamw \
    --lr 1e-3 \
    --num_epochs 150 \
    --output_dir ./Output/VGG11/Conv2d
```

### Train VGG11 with Pure ConvNN

```shell
python main.py \
    --model vgg11 \
    --layer ConvNN \
    --K 9 \
    --sampling_type spatial \
    --num_samples 8 \
    --similarity_type Col \
    --aggregation_type Col \
    --dataset cifar10 \
    --optimizer adamw \
    --lr 1e-3 \
    --num_epochs 150 \
    --output_dir ./Output/VGG11/ConvNN
```

### Train VGG11 with Branching Layer (Hybrid Conv2d + ConvNN)

```shell
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 3 \
    --K 9 \
    --padding 1 \
    --branch_ratio 0.5 \
    --sampling_type all \
    --num_samples -1 \
    --similarity_type Col \
    --aggregation_type Col \
    --dataset cifar10 \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr 1e-4 \
    --scheduler cosine \
    --num_epochs 150 \
    --output_dir ./Output/VGG11/Branching_50
```

### Train ResNet18 with ConvNN + Attention

```shell
python main.py \
    --model resnet18 \
    --layer ConvNN_Attn \
    --K 9 \
    --attention_dropout 0.1 \
    --similarity_type Loc_Col \
    --aggregation_type Loc_Col \
    --lambda_param 0.5 \
    --dataset cifar100 \
    --optimizer adamw \
    --lr 1e-3 \
    --num_epochs 150 \
    --use_amp \
    --output_dir ./Output/ResNet18/ConvNN_Attn
```

### Train with Mixed Precision and Compiled Model

```shell
python main.py \
    --model vgg11 \
    --layer Branching \
    --branch_ratio 0.5 \
    --dataset cifar10 \
    --optimizer adamw \
    --lr 1e-4 \
    --num_epochs 150 \
    --use_amp \
    --use_compiled \
    --compile_mode reduce-overhead \
    --output_dir ./Output/VGG11/Branching_Compiled
```

### Test Mode Only

```shell
python main.py \
    --model vgg11 \
    --layer ConvNN \
    --test_only \
    --device cuda
```

## Output Files

After running training, the following files are saved in `--output_dir`:

- **`args.txt`**: All command-line arguments used for the experiment
- **`model.txt`**: Model architecture and parameter summary
- **`train_eval_results.txt`**: Training and evaluation results (loss, accuracy per epoch)

## Supported Architectures & Layers

### Models
- **VGG**: `vgg11`, `vgg13`, `vgg16`, `vgg19`
- **ResNet**: `resnet18`, `resnet34`

### Layers
- **`Conv2d`**: Standard 2D convolution
- **`ConvNN`**: 2D K-nearest neighbor convolution
- **`ConvNN_Attn`**: ConvNN with attention (Q, K, V projections)
- **`Branching`**: Parallel branches of Conv2d and ConvNN
- **`Branching_Attn`**: Parallel branches with attention
- **`Conv2d_New`**: Alternative Conv2d implementation

## Notes

- Set `--num_samples -1` with `--sampling_type all` to use all spatial locations as candidates
- `--branch_ratio 0.0` means 100% Conv2d (baseline)
- `--branch_ratio 1.0` means 100% ConvNN
- `--branch_ratio 0.5` means 50% Conv2d and 50% ConvNN
- Use `--use_compiled` for significant speedups on long training runs (first epoch slower due to compilation)
- Mixed precision (`--use_amp`) can reduce memory usage by ~50% with minimal accuracy impact

## Project Structure

```
.
├── models/
│   ├── vgg.py              # VGG architecture implementation
│   ├── resnet.py           # ResNet architecture implementation
│   ├── layers2d.py         # 2D layers for CNN models
│   └── layers1d.py         # 1D layers for CNN models
├── Project Docs/           # Project documentation, reports, posters, etc.
├── Data/                   # Dataset directory (CIFAR-10/100, ImageNet) 
├── dataset.py              # CIFAR-10/100 & ImageNet wrappers
├── train_eval.py           # Training & evaluation loop
├── main.py                 # CLI entrypoint for training
├── utils.py                # I/O, logging, seed setup
├── requirements.txt        # Python dependencies
├── README.md               # ← you are here
└── LICENSE                 # MIT License
```

## License

Convolutional-Nearest-Neighbor is released under the MIT License. Please see the [LICENSE](https://www.google.com/search?q=LICENSE) file for more information.

## Contributing

Contributions, issues, and feature requests are welcome\!
Please reach out to:

  - **Mingi Kang** [mkang2@bowdoin.edu](mailto:mkang2@bowdoin.edu)
  - **Jeova Farias** [j.farias@bowdoin.edu](mailto:j.farias@bowdoin.edu)

<!-- end list -->
