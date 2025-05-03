# ConvNN: Convolutional Nearest Neighbor for Neural Networks

## Grants/Funding
- Christenfeld Summer Research Fellowship, **Bowdoin College**
- Google AI 2024 Funding, **Last Mile Fund**
- NYC Stem Funding, **Last Mile Fund**

### Bowdoin Summer 2024, Spring 2025, Summer 2025

## Overview 
ConvNN introduces a novel convolution technique for computer vision, focusing on the relationships between center pixels and their nearest neighbors, rather than just adjacent pixels. This approach aims to improve classification accuracy by considering a broader context within images.

### Key Concepts
- **Nearest Neighbor Convolution**: Replaces traditional pixel adjacency with the closest K-nearest pixels, offering a new way to analyze image data.
- **Flexibility**: ConvNN is adaptable to both 1D and 2D data, and can be easily integrated into existing neural networks using custom PyTorch modules.

### Implementation
ConvNN layers are implemented using the `torch.nn.Module` class:
- **Conv1dNN**: Handles 1D convolution with nearest neighbors.
- **Conv2dNN**: Applies nearest neighbor convolution in 2D.
- **Spatial Variants**: Includes spatial considerations for both 1D and 2D layers.
- **Pixel Shuffle Layers**: Custom layers for pixel shuffling and unshuffling in 1D.

A [Colab Notebook]() is available for hands-on experimentation.


## Usage
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
python main.py --model ConvNN --num_layers 2 --hidden_dim 8 --k_kernel 5 --sampling Random --num_samples 16 --shuffle_pattern BA --shuffle_scale 2 --batch_size 16 --output_dir ./Output/ConvNN --device mps --num_epochs 100 -lr 0.0001 --dataset CIFAR100
```

## License
Convolutional-Nearest-Neighbor is released under the MIT License. Please see the [LICENSE](LICENSE) file for more information.

## Contributing
We welcome feedback and collaboration! Please reach out to: 
- Mingi Kang : mkang2@bowdoin.edu 
- Jeova Farias : j.farias@bowdoin.edu
