'''MNIST 2D data for training 2D CNN Models'''

import torch 
import torch.nn as nn
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt 


### Classification Data - MNIST###
class MNIST: 
   def __init__(self, batch_size=64): 
      self.batch_size = batch_size
      
      self.train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
      self.test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
      self.train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True)
      self.test_loader = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False)      

   def shape(self): 
      return self.train_data[0][0].shape
   
   def visual(self): 
      plt.figure(figsize=(6, 3)) 
      plt.imshow(self.train_data[0][0].squeeze(), cmap='gray')
      plt.show()
      
### Denoising Data - NoisyMNIST###
class NoisyMNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, noise_std=0.3):
        super(NoisyMNIST, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.noise_std = noise_std

    def __getitem__(self, index):
        img, target = super(NoisyMNIST, self).__getitem__(index)
        noisy_img = img + self.noise_std * torch.randn_like(img)
        return noisy_img, img, target

class MNIST_denoise:
    def __init__(self, batch_size=64, noise_std=0.3):
        self.batch_size = batch_size
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.train_data = NoisyMNIST(root='./data', train=True, download=True, transform=transform, noise_std=noise_std)
        self.test_data = NoisyMNIST(root='./data', train=False, download=True, transform=transform, noise_std=noise_std)
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False)
      
    def shape(self):
        return self.train_data[0][0].shape
   
    def visual(self):
        noisy_img, img, _ = self.test_data[0]
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img.squeeze(), cmap='gray')
        
        plt.subplot(1, 2, 2)
        plt.title("Noisy Image")
        plt.imshow(noisy_img.squeeze(), cmap='gray')
        plt.show()
        

def test_denoise_visual(model, test_loader):
    
    count = 0
    for test_data in test_loader:
        noisy_img, img = test_data[0], test_data[1]
        
        # Select the first image in the batch
        noisy_img = noisy_img[0]
        img = img[0]

        output_img = model(noisy_img.unsqueeze(0).to('mps'))  # Add batch dimension back for the model

        plt.subplot(1, 3, 1)
        plt.title('Noisy image')
        plt.imshow(noisy_img.squeeze().cpu().numpy(), cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title('Denoised image')
        plt.imshow(output_img.squeeze().detach().cpu().numpy(), cmap='gray')
        
        plt.subplot(1, 3, 3)
        plt.title('Original image')
        plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')

        plt.show()
        if count == 3: 
            break
        else:
            count += 1
            # Remove this break to visualize more images

### EXAMPLE USAGE ### 
# mnist = MNIST()

# mnist_denoise = MNIST_denoise(noise_std=0.5)
# print(mnist_denoise.shape())
# mnist_denoise.visual()