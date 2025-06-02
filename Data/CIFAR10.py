'''CIFAR10 2D data for training 2D CNN Models'''

import torch 
import torch.nn as nn
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt 

### Classification Data - CIFAR10 ###
class CIFAR10: 
   def __init__(self, batch_size=64):
      self.batch_size = batch_size
      
      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
      ])
      
      self.train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
      self.test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
      self.train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True)
      self.test_loader = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False)
      
   def shape(self): 
      return self.train_data[0][0].shape
   
   def visual(self): 
      img = self.test_data[0][0].permute(1, 2, 0)  # Change the order of dimensions for displaying
      img = img * torch.tensor([0.2023, 0.1994, 0.2010]) + torch.tensor([0.4914, 0.4822, 0.4465])  # Unnormalize
      plt.figure(figsize=(6, 3)) 
      plt.imshow(img)
      plt.show()      
        

### Denoising Data - NoisyCIFAR10###
class NoisyCIFAR10(datasets.CIFAR10):
   def __init__(self, 
                root, 
                train=True, 
                transform=None, 
                target_transform=None, 
                download=False, 
                noise_std=0.3
                ):
       
        super(NoisyCIFAR10, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.noise_std = noise_std

   def __getitem__(self, index):
        img, target = super(NoisyCIFAR10, self).__getitem__(index)
        
        # Add Gaussian noise to the image
        noisy_img = img + self.noise_std * torch.randn_like(img)
        return noisy_img, img, target

class CIFAR10_denoise:
    def __init__(self, 
                 batch_size=64, 
                 noise_std=0.3):
        
        self.batch_size = batch_size
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.train_data = NoisyCIFAR10(root='./data', train=True, download=True, transform=transform, noise_std=noise_std)
        self.test_data = NoisyCIFAR10(root='./data', train=False, download=True, transform=transform, noise_std=noise_std)
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False)
      
    def shape(self):
        return self.train_data[0][0].shape
   
    def visual(self):
        noisy_img, img, _ = self.test_data[0]
        img = img.permute(1, 2, 0)
        img = img * torch.tensor([0.2023, 0.1994, 0.2010]) + torch.tensor([0.4914, 0.4822, 0.4465])
        
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(img)

        
        noisy_img = noisy_img.permute(1, 2, 0)
        noisy_img = noisy_img * torch.tensor([0.2023, 0.1994, 0.2010]) + torch.tensor([0.4914, 0.4822, 0.4465])
        plt.subplot(1, 2, 2)
        plt.title('Noisy Image')
        plt.imshow(noisy_img)
        
        plt.show()


def test_denoise_visual(model, test_loader):
    count = 0
    
    for test_data in test_loader:
        noisy_img, img = test_data[0], test_data[1]
        
        # Select the first image in the batch
        noisy_img = noisy_img[0]
        img = img[0]

        # Move tensors to MPS device
        noisy_img = noisy_img.to('mps')
        img = img.to('mps')

        # Get the model output
        output_img = model(noisy_img.unsqueeze(0))

        # Move tensors back to CPU and detach
        noisy_img = noisy_img.cpu().detach()
        output_img = output_img.squeeze(0).cpu().detach()
        img = img.cpu().detach()

        # Permute dimensions to (height, width, channels) for matplotlib
        noisy_img = noisy_img.permute(1, 2, 0)
        output_img = output_img.permute(1, 2, 0)
        img = img.permute(1, 2, 0)

        # Denormalize the images (assuming CIFAR-10 normalization)
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])
        noisy_img = noisy_img * std + mean
        output_img = output_img * std + mean
        img = img * std + mean

        plt.figure(figsize=(15, 5))  # Adjust the figure size if needed

        plt.subplot(1, 3, 1)
        plt.title('Noisy image')
        plt.imshow(noisy_img.numpy())

        plt.subplot(1, 3, 2)
        plt.title('Denoised image')
        plt.imshow(output_img.numpy())
        
        plt.subplot(1, 3, 3)
        plt.title('Original image')
        plt.imshow(img.numpy())

        plt.show()

        if count == 3: 
            break
        else:
            count += 1    
### EXAMPLE USAGE ### 
#cifar10 = CIFAR10()

# cifar10_noise = CIFAR10_denoise(noise_std=0.5)
# print(cifar10_noise.shape())
# cifar10_noise.visual()