# Convolutional Nearest Neighbor Attention (ConvNN-Attention) for Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview
Convolutional Nearest Neighbor Attention (ConvNN-Attention) is a novel attention mechanism featuring hard selection of nearest neighbors for Vision Transformers (ViTs). Traditional attention compute similarities across all token with soft selection, while ConvNN-Attention focuses on the $k$ most relevant neighbors with convolutional operations to aggregate information, enhancing efficiency and potentially improving performance.

### Key Concepts
- **Vision Transformer (ViT)**: A standard ViT architecture where the multi-head self-attention block can be replaced with our custom `MultiHeadConvNNAttention` and branching blocks.
- **Layer Flexibility**: The project supports single-block types (e.g., `MultiHeadConvNNAttention` only) and branching architectures (e.g., `BranchingAttention`, `BranchingConv`) to combine the strengths of different operations.
- **Sampling Strategies**: To manage computational complexity, `ConvNN` layers support sampling strategies: `all` (dense), `random`, and `spatial`.
- **Number of Heads**: `MultiHeadConvNNAttention` does not use head splitting. Each block processes the full feature dimension.

### Implementation

**ViT (`vit.py`)**: ViT implementation with modular attention layers.

**Attention Layers (`vit_layers.py`)**: Implements various attention mechanisms:
- **`MultiHeadAttention`**: Standard Transformer multi-head self-attention
- **`MultiHeadConvNNAttention`**: Convolutional Nearest Neighbor Attention layer with k-NN selection
- **`MultiHeadKvtAttention`**: An implementation of **k-NN Attention for boosting Vision Transformers**
- **`MultiHeadLocalAttention`**: Local Attention implementation from **lucidrains**
- **`MultiHeadSparseAttention`**: Sparse Attention implementation from **Generating Long Sequences with Sparse Transformers**


## Installation

Install the required dependencies:

```shell
pip install -r requirements.txt
```

## Command-Line Interface

### Main Script (`vit_main.py`)

Run `python vit_main.py --help` to see all available options.

#### Model & Layer Configuration

| Flag                  | Default     | Choices                                                                    | Description                                     |
| --------------------- | ----------- | -------------------------------------------------------------------------- | ----------------------------------------------- |
| `--layer`             | `Attention` | `Attention`, `ConvNNAttention`, `KvtAttention`, `LocalAttention`, `SparseAttention` | Layer type for ViT transformer blocks.          |
| `--patch_size`        | `16`        | *integer*                                                                  | Patch size for attention models.                |
| `--num_layers`        | `12`        | *integer*                                                                  | Number of transformer encoder layers.           |
| `--num_heads`         | `3`         | *integer*                                                                  | Number of attention heads.                      |
| `--d_hidden`          | `192`       | *integer*                                                                  | Hidden dimension for the model.                 |
| `--d_mlp`             | `768`       | *integer*                                                                  | MLP dimension for the model.                    |
| `--dropout`           | `0.1`       | *float*                                                                    | Dropout rate for linear layers.                 |
| `--attention_dropout` | `0.1`       | *float*                                                                    | Dropout rate for attention layers.              |

#### ConvNN-Attention Specific Parameters

| Flag                  | Default        | Choices                           | Description                                         |
| --------------------- | -------------- | --------------------------------- | --------------------------------------------------- |
| `--convolution_type`  | `depthwise`     | `standard`, `depthwise`, `depthwise-separable` | Convolution type for ConvNN-Attention layers.       |
| `--softmax_topk_val`  | `True`         | `True`, `False`                   | Use softmax on top-k values.           |
| `--K`                 | `9`            | *integer*                         | Number of nearest neighbors (k-NN) or kernel size.  |
| `--sampling_type`     | `all`          | `all`, `random`, `spatial`        | Sampling strategy for neighbor candidates.          |
| `--num_samples`       | `-1`           | *integer*                         | Number of samples for `random`/`spatial` modes. `-1` for all. |
| `--sample_padding`    | `0`            | *integer*                         | Padding for spatial sampling.                       |
| `--magnitude_type`    | `matmul`       | `cosine`, `euclidean`, `matmul`   | Similarity metric for nearest neighbors.            |
| `--coordinate_encoding` | `False`      | `True`, `False`                   | Enable coordinate encoding in ConvNN-Attention layers.        |

- (note) For `random` sampling type, set `--num_samples` to the desired number of neighbors to sample (e.g., `4`). For `spatial` sampling type, set `--num_samples` to the spatially separated grid (e.g., `3` for 3x3). For dense attention, set `--sampling_type` to `all` and `--num_samples` to `-1`.

#### Convolution-Specific Parameters

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--kernel_size` | `9` | *integer* | kernel size for convolution 1D |


#### Dataset Configuration

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--dataset` | `cifar10` | `cifar10`, `cifar100`, `imagenet` | Dataset to use for training and evaluation |
| `--data_path` | `./Data` | *path* | Path to the dataset directory |
| `--resize` | `224` | *integer* | Resize images to specified size (e.g., 224 for 224x224) |

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
| `--lr` | `1e-4` | *float* | Initial learning rate |
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

### Train ViT-Tiny with Standard self-Attention

```shell
python vit_main.py \
    --layer Attention \
    --num_layers 12 \
    --num_heads 3 \
    --d_hidden 192 \
    --d_mlp 768 \
    --dataset cifar10 \
    --optimizer adamw \
    --lr 1e-3 \
    --num_epochs 150 \
    --output_dir ./Output/ViT/Attention
```

### Train ViT-Tiny with Pure ConvNN

```shell
python vit_main.py \
    --layer ConvNNAttention \
    --convolution_type depthwise \
    --num_layers 12 \
    --d_hidden 192 \
    --d_mlp 768 \
    --K 9 \
    --sampling_type spatial \
    --num_samples 8 \
    --similarity_type Col \
    --aggregation_type Col \
    --dataset cifar10 \
    --optimizer adamw \
    --lr 1e-3 \
    --num_epochs 150 \
    --output_dir ./Output/ViT/ConvNNAttention
```

### Test Mode Only

```shell
python vit_main.py \
    --layer Attention \
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
- **ViT**: Vision Transformer with modular attention layers

### Layers
- **`MultiHeadAttention`**: Standard multi-head self-attention
- **`MultiHeadConvNNAttention`**: Convolutional Nearest Neighbor Attention
- **`MultiHeadKvtAttention`**: k-NN Attention for Vision Transformers
- **`MultiHeadLocalAttention`**: Local Attention from lucidrains
- **`MultiHeadSparseAttention`**: Sparse Attention from Sparse Transformers


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
│   ├── vit.py              # ViT architecture implementation
│   └── vit_layers.py       #  layers for ViT models
├── Data/                   # Dataset directory (CIFAR-10/100) 
├── dataset.py              # CIFAR-10/100 wrappers
├── train_eval.py           # Training & evaluation loop
├── vit_main.py             # CLI entrypoint for training
├── utils.py                # I/O, logging, seed setup
├── requirements.txt        # Python dependencies
├── README_VIT.md           # ← you are here
└── LICENSE                 # MIT License
```

## License

Convolutional-Nearest-Neighbor is released under the MIT License. Please see the [LICENSE](https://www.google.com/search?q=LICENSE) file for more information.

