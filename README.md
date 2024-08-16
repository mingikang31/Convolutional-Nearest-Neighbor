# ConvNN: Convolutional Nearest Neighbor for Neural Networks

### Christenfeld Summer Research Fellowship - Mingi Kang & Jeova Farias 
### Bowdoin Summer Research 2024 


![DETR](https://cdn.prod.website-files.com/614c82ed388d53640613982e/646371e3bdc5ca90dee5331b_convolutional-neural-network%20(1).webp)
*Picture is from SuperAnnotate: Convolutional Neural Networks: 1998-2023 Overview


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

A [Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb) is available for hands-on experimentation.


### Usage Example
```Python 
Conv2d_NN(in_channels, out_channels, K, stride, padding, shuffle_pattern, shuffle_scale, samples, magnitude_type)
```
- **K**: The number of nearest neighbors to consider. 
- **K** and **stride** must be the same value. 
- **shuffle_pattern** can be `'N/A', 'B', 'A',` or `'BA'` (Not Applicable, Before, After, Before + After).
- **samples** can be `'all'` or an integer value representing the number of samples you would like to consider. 
- **magnitude_type** can be `'distance'` or `'similarity'`. 

# Classification Testing
**Model/Training Information** 
- **Batch Size**: 64
- **Epochs**: 10
- **Loss Function**: `'nn.CrossEntropyLoss()'`
- **Ex.** K = 8 (8 Nearest Neighbors)

We have tested various models for each dataset to compare their performance. 

## I. MNIST Classification
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Parameters</th>
      <th>Ave Epoch Time</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Conv2dNN (all samples)</td>
      <td>189,590</td>
      <td>48.885s</td>
      <td>94.74%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Conv2dNN N samples (10 samples)</td>
      <td>189,590</td>
      <td>50.306s</td>
      <td>95.73%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Conv2dNN Spatial (9 samples)</td>
      <td>158,885</td>
      <td>87.828s</td>
      <td>11.35%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CNN Model 1</td>
      <td>99,140</td>
      <td>6.481s</td>
      <td>98.45%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CNN Model 2</td>
      <td>716,706</td>
      <td>8.348s</td>
      <td>99.26%</td>
    </tr>
  </tbody>
</table>


## II. Fashion MNIST Classification
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Parameters</th>
      <th>Ave Epoch Time</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Conv2dNN (all samples)</td>
      <td>189,590</td>
      <td>45.583</td>
      <td>86.95%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Conv2dNN N samples (10 samples)</td>
      <td>189,590</td>
      <td>49.318s</td>
      <td>86.53%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Conv2dNN Spatial (9 samples)</td>
      <td>158,885</td>
      <td>90.273s</td>
      <td>16.17%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CNN Model 1</td>
      <td>99,140</td>
      <td>6.509s</td>
      <td>90.58%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CNN Model 2</td>
      <td>716,706</td>
      <td>8.378s</td>
      <td>91.88%</td>
    </tr>
  </tbody>
</table>



## III. CIFAR10 Classification
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Parameters</th>
      <th>Ave Epoch Time</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Conv2dNN (all samples)</td>
      <td>238,870</td>
      <td>44.336s</td>
      <td>54.06%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Conv2dNN N samples (10 samples)</td>
      <td>238,870</td>
      <td>55.509s</td>
      <td>57.15%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Conv2dNN Spatial (9 samples)</td>
      <td>206,965</td>
      <td>107.976s</td>
      <td>20.93%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CNN Model 1</td>
      <td>137,630</td>
      <td>7.608s</td>
      <td>59.35%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CNN Model 2</td>
      <td>999,458</td>
      <td>11.745s</td>
      <td>71.41%</td>
    </tr>
  </tbody>
</table>

MNIST, Fashion MNIST, CIFAR10 training/testing Classification file can be found in this [Notebook](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918).


# Denoising Testing
### Model/Training Information
- **Batch Size**: 64
- **Epochs**: 10
- **Loss Function**: `nn.MSELoss()`
- **Evaluation Metric**: PSNR (Peak Signal-to-Noise Ratio)
- **Ex.** K = 8 (8 Number of Neighbors)

We have tested various models for each dataset to compare their performance. 

## I. MNIST Denoising
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Parameters</th>
      <th>Ave Epoch Time</th>
      <th>Ave Loss</th>
      <th>Ave PSNR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Conv2dNN (all samples)</td>
      <td>35,344</td>
      <td>63.513s</td>
      <td>0.018</td>
      <td>17.489</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Conv2dNN N samples (10 samples)</td>
      <td>35,344</td>
      <td>70.132s</td>
      <td>0.013</td>
      <td>18.814</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Conv2dNN Spatial (9 samples)</td>
      <td>2,236</td>
      <td>140.166s</td>
      <td>0.097</td>
      <td>10.191</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CNN Model 1</td>
      <td>2,511</td>
      <td>8.102s</td>
      <td>0.005</td>
      <td>22.828</td>
    </tr>

  </tbody>
</table>


## II. Fashion MNIST Denoising
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Parameters</th>
      <th>Ave Epoch Time</th>
      <th>Ave Loss</th>
      <th>Ave PSNR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Conv2dNN (all samples)</td>
      <td>35,344</td>
      <td>63.953s</td>
      <td>0.583</td>
      <td>2.354</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Conv2dNN N samples (10 samples)</td>
      <td>35,344</td>
      <td>66.094s</td>
      <td>0.628</td>
      <td>2.020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Conv2dNN Spatial (9 samples)</td>
      <td>2,236</td>
      <td>123.934s</td>
      <td>0.679</td>
      <td>1.684</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CNN Model 1</td>
      <td>2,511</td>
      <td>8.469s</td>
      <td>0.074</td>
      <td>11.329</td>
    </tr>
  </tbody>
</table>

## III. CIFAR10 Denoising
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Parameters</th>
      <th>Ave Epoch Time</th>
      <th>Ave Loss</th>
      <th>Ave PSNR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Conv2dNN (all samples)</td>
      <td>41,792</td>
      <td>65.307s</td>
      <td>0.937</td>
      <td>0.293</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Conv2dNN N samples (10 samples)</td>
      <td>41,792</td>
      <td>83.208s</td>
      <td>1.438</td>
      <td>1.465</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Conv2dNN Spatial (9 samples)</td>
      <td>2,638</td>
      <td>149.808s</td>
      <td>1.438</td>
      <td>-1.572</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CNN Model 1</td>
      <td>2,963</td>
      <td>10.016s</td>
      <td>0.498</td>
      <td>3.043</td>
    </tr>

  </tbody>
</table>

MNIST, Fashion MNIST, CIFAR10 training/testing Denosing results can be found in this [Noteboook](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918).


# K Testing
### Model/Training Information
- **Batch Size**: 64
- **Epochs**: 10
- **Loss Function**: `nn.CrossEntropyLoss()`
- **Ex.** K = 8 (8 neighbors)

We are testing and evaluating the impact of the number of neighbors in **Convolution Nearest Neighbors**. For instance, if K = 10, we consider the 10 most similar pixels to the center pixel. The models we used for these tests consider all samples.


## I. MNIST Classification K Test
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Parameters</th>
      <th>Ave Epoch Time</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Conv2dNN K=3</td>
      <td>169,190</td>
      <td>39.489s</td>
      <td>94.96%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Conv2dNN K=5</td>
      <td>177,350</td>
      <td>42.009s</td>
      <td>94.89%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Conv2dNN K=7</td>
      <td>185,510</td>
      <td>47.788s</td>
      <td>94.81%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Conv2dNN K=9</td>
      <td>193,670</td>
      <td>47.788s</td>
      <td>94.61%</td>
    </tr>
  </tbody>
</table>


## II. Fashion MNIST Classification K Test
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Parameters</th>
      <th>Ave Epoch Time</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Conv2dNN K=3</td>
      <td>169,190</td>
      <td>37.415s</td>
      <td>87.15%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Conv2dNN K=5</td>
      <td>177,350</td>
      <td>40.225s</td>
      <td>87.87%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Conv2dNN K=7</td>
      <td>185,510</td>
      <td>43.213s</td>
      <td>87.98%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Conv2dNN K=9</td>
      <td>193,670</td>
      <td>47.199s</td>
      <td>87.12%</td>
    </tr>
  </tbody>
</table>


## III. CIFAR10 Classification K Test
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Parameters</th>
      <th>Ave Epoch Time</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Conv2dNN K=3</td>
      <td>217,670</td>
      <td>38.885s</td>
      <td>54.98%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Conv2dNN K=5</td>
      <td>226,150</td>
      <td>39.271s</td>
      <td>53.83%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Conv2dNN K=7</td>
      <td>234,630</td>
      <td>41.470s</td>
      <td>54.05%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Conv2dNN K=9</td>
      <td>243,110</td>
      <td>97.124s</td>
      <td>54.16%</td>
    </tr>
  </tbody>
</table>

MNIST, Fashion MNIST, CIFAR10 Classification K Test training/testing/models can be found in this [Noteboook](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918).

# Notebooks

* [ConvNN 2D & ConvNN 1D](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb): 
In this notebook, we demonstrate the functionality and parameters for **Convolution Nearest Neighbor** in 2D and 1D. 

* [Classification ConvNN](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb): In this notebook, we demonstrate how to implement a simple neural network using **Convolution Nearest Neighbor** in 2D and 1D with training/testing of simple data for classification (ie. MNIST, FashionMNIST, etc)

* [Denoising ConvNN](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb): In this notebook, we demonstrate how to implement a simple neural network using **Convolution Nearest Neighbor** in 2D and 1D with training/testing of simple data for Denoising (ie. MNIST, FashionMNIST, etc)

* [Panoptic Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb): Demonstrates how to use DETR for panoptic segmentation and plot the predictions.


# Future Work
Our ongoing research focuses on: 
1. Improving learning performance for spatial Conv2dNN 
2. Exploring hybrid branching networks with traditional convolution and ConvNN
3. Addressing training time inconsistencies 
4. Testing on larger images and implementing Faiss for faster top-k operations in PyTorch. 

# License
DETR is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

# Contributing
We Welcome feedback and collaboration. Please reach out to: 
- Mingi Kang : mkang2@bowdoin.edu 
- Jeova Farias : j.farias@bowdoin.edu
