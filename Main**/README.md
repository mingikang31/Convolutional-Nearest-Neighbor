# Convolutional Nearest Neighbor (ConvNN)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Grants & Funding
- **Christenfeld Summer Research Fellowship**, Bowdoin College
- **Google AI 2024 Funding**, Last Mile Fund
- **NYC Stem Funding**, Last Mile Fund

**Project periods:** Bowdoin Summer 2024, Spring 2025, Summer 2025

## Overview 
Traditional convolutions only see a fixed grid of adjacent pixels. **ConvNN**: 
- Finds the _K_-nearest neighbors of each token (pixel or feature vector). 
- Aggregates their values by distance- or simlarity-based weightings. 
- Captures long-range dependencies and relationships between pixels.
- Is drop-in compatible with both 1D and 2D PyTorch modules. 

### Key Concepts
- **Convolutional Nearest Neighbor (ConvNN)**: A new convolutional layer that replaces traditional pixel adjacency with the closest K-nearest pixels.
- **Flexibility**: ConvNN is adaptable to both 1D and 2D data, and can be easily integrated into existing neural networks using custom PyTorch modules.
- **Attention Mechanism**: ConvNN can be combined with attention mechanisms to enhance feature extraction and representation learning.
- **Spatial Variants**: ConvNN includes spatial variants for both 1D and 2D layers, allowing for more flexible and efficient processing of spatial data.

### Implementation 
ConvNN layers are implemented using the `torch.nn.Module` class:
- **Conv1d_NN**: Handles 1D convolution with nearest neighbors.
- **Conv2d_NN**: Applies nearest neighbor convolution in 2D.
- **Conv2d_NN_Attn**: Incorporates attention mechanism's linear layers for enhanced feature extraction.
- **Pixel Shuffle Layers**: Custom layers for pixel shuffling and unshuffling in 1D.

## Installation
```Shell 
git clone https://github.com/mingikang31/Convolutional-Nearest-Neighbor.git
```
Then, Install PyTorch and torchvision
```Shell
pip install torch torchvision
```
### Training & Evaluation 
To train and evaluate ConvNN on CIFAR100 for 100 epochs run, use this command: 

```Shell 
python main.py --model Simple --layer ConvNN --num_layers 2 --hidden_dim 8 --K 5 --sampling Random --num_samples 64 --shuffle_pattern BA --shuffle_scale 2 --batch_size 64 --output_dir ./Output/ConvNN --device mps --num_epochs 100 --lr 0.0001 --dataset CIFAR100
```
- Simple model with ConvNN Layers, 2 layers,

## Command-Line Interface 
All flags are defined in the `main.py` file. Run `python main.py --help` to see all available options.

### Model & Layer 
| Flag                 | Default       | Choices                   | Description                                  |
| -------------------- | ------------- | ------------------------- | -------------------------------------------- |
| `--model`            | `Simple`      | `Simple`, `VGG`, `ViT`    | Base architecture                            |
| `--layer`            | `ConvNN`      | `Conv2d`, `ConvNN`, `ConvNN_Attn`, `Attention`, `Conv1d`, and their paired combos| Which convolution/attention layer to insert                        |
| `--num_layers`       | `4`           | _integer_                 | Number of sequential layers                  |
| `--hidden_dim`       | `16`          | _integer_                 | Width of hidden feature maps                 |


Notes: 


### ConvNN-Specific 
| Flag                 | Default       | Choices                   | Description                                  |
| -------------------- | ------------- | ------------------------- | -------------------------------------------- |
| `--K`                | `9`           | _integer_                 | Number of nearest neighbors                  |
| `--sampling`         | `All`         | `All`, `Random`, `Spatial`| How to select neighbors                      |
| `--num_samples`      | `64`          | _integer_ or `all`        | Max samples per query (overrides `All`)      |
| `--shuffle_pattern`  | `BA`          | `BA`, `NA`                | Before-After vs no shuffle in 2D             |
| `--shuffle_scale`    | `2`           | _integer_                 | Pixel unshuffle/downscale factor             |
| `--magnitude_type`   | `similarity`  | `similarity`, `distance`  | Weighting type                               |
| `--location_channels`| _flag_        |                           | Append XY-coordinate channels (boolean flag) |

### Attention-Specific 
| Flag                 | Default       | Description                                      |
| -------------------- | ------------- | ------------------------------------------------ |
| `--num_heads`        | `4`           | Number of attention heads (Simple, VGG, and ViT) |
| `--num_patches`      | `4`           | Number of patches (ViT)                          |
| `--patch_size`       | `16`          | Patch size (ViT)                                 |
| `--d_model`          | `9`           | Dimension of the transformer embedding (ViT)     |

### Data & Training 
| Flag                 | Default        | Choices                   | Description                                  |
| -------------------- | -------------- | ------------------------- | -------------------------------------------- |
| `--dataset`          | `cifar100`     | `cifar10`, `cifar100`, `imagenet`| Dataset for training and evaluation   |
| `--batch_size`       | `64`           | _integer_                 | Batch size for training                      |
| `--output_dir`       | `./Output/Out` | _string_                  | Directory for saving the output              |
| `--num_epochs`       | `100`          | _integer_                 | Number of epochs for training                |
| `--use_amp`          | flag           | _float_                   | Enable mixed-precision (FP16)                |
| `--clip_grad_norm`   | None           | _float_                   | Maximum gradient norm (if any)               |


### Optimization & LR
| Flag                 | Default        | Choices                     | Description                                  |
| -------------------- | -------------- | --------------------------- | -------------------------------------------- |
| `--criterion`        | `CrossEntropy` | `CrossEntropy`, `MSE`       | Loss function                                |
| `--optimizer`        | `adamw`        | `adam`, `sgd`, `adamw`      | Optimizer                                    |
| `--momentum`         | `0.9`          | _float_                     | Only for SGD                                 |  
| `--weight_decay`     | `1e-6`         | _float_                     | L2 regularlization                           |
| `--lr`               | `1e-3`         | _float_                     | Base learning rate                           |
| `--scheduler`        | `step`         | `step`, `cosine`, `plateau` | LR schedule type                             |
| `--lr_step`          | `20`           | _float_                     | step scheduler step size                     |
| `--lr_gamma`         | `0.1`          | _float_                     | step scheduler decay factor                  |


### Device & Reproduction
| Flag                 | Default        | Choices                     | Description                                  |
| -------------------- | -------------- | --------------------------- | -------------------------------------------- |
| `--device`           | `cuda`         | `cuda`, `mps`, `cpu`        | Device for training inference                |
| `--seed`             | `0`            | _integer_                   | Random seed for reproducibility              |
| `--output_dir`       | `./Output/Out` | _path_                      | path to save output                          |  


## Project Structure 
```Shell
.
├── dataset.py            # CIFAR-10/100 & ImageNet wrappers
├── train_eval.py         # Training & evaluation loop
├── simple.py             # SimpleModel w/ interchangeable layers
├── vgg.py                # VGG backbone with ConvNN support
├── vit.py                # ViT backbone & patch embedding
├── utils.py              # I/O, logging, seed setup
├── main.py               # CLI entrypoint (see above)
├── README.md             # ← you are here
└── LICENSE               # MIT License
```


## License 
Convolutional-Nearest-Neighbor is released under the MIT License. Please see the [LICENSE](LICENSE) file for more information.

## Contributing 
Contributions, issues, and feature requests are welcome! 
Please reach out to: 
- **Mingi Kang** <mkang2@bowdoin.edu>
- **Jeova Farias** <j.farias@bowdoin.edu>
