# Torch
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim 

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random



from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

class NoisyCBSD68(Dataset):
    def __init__(self, 
                 target_count=200,
                 noise_std=0.3
                 ):
        
        super(NoisyCBSD68, self).__init__()
        self.noise_std = noise_std
        
        # self.images = self.load_images("/Users/mingikang/Developer/Convolutional-Nearest-Neighbor/Data/CBSD68_data")
        self.images = self.load_images("/home/mkang2/Convolutional-Nearest-Neighbor/Data/CBSD68_data")

        self.data = self.create_image_set(self.images, target_count)
        
    def __getitem__(self, index):
        img, target = self.data[index], self.data[index]
        
        # # Add noise v1
        # noisy_img = img + self.noise_std * torch.randn_like(img)
        # noisy_img = torch.clamp(noisy_img, 0, 1)
        
        ####
        
        # # Add Noise v2
        # add_gaussian_noise = transforms.GaussianBlur(sigma=self.noise_std)
        # noisy_img = add_gaussian_noise(img)
        
        # # Add Noise v3
        # blurrer = v2.GaussianBlur(kernel_size=(15, 15), sigma=(self.noise_std))
        # noisy_img = blurrer(img)
        
        
        # # Add Noise v4
        noise = torch.randn_like(img) * self.noise_std  # Direct scaling without division by 255
        noisy_img = img + noise        
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
        
        return noisy_img, img, target
    

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def load_images(directory):
        images = []
        
        for filename in os.listdir(directory):
            try: 
                img = Image.open(os.path.join(directory, filename)).convert('RGB')  
                img_array = np.array(img) / 255.0  
                img_array = torch.from_numpy(img_array).float()
                img_array = img_array.permute(2, 0, 1)

                images.append(img_array)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
            
        return images

    @staticmethod
    def create_image_set(images, target_count, target_size=200):
        
        image_set = []
        
        for i in range(target_count):
            if not images:  # Check if images list is empty
                raise ValueError("No images were successfully loaded")
                
            img = random.choice(images)
            # No need to unsqueeze - the tensor is already in (C,H,W) format
            
            transform_list = [transforms.RandomCrop(target_size)]
            
            if random.random() > 0.3:  # 30% chance of flipping
                transform_list.append(transforms.RandomHorizontalFlip(p=1.0))
            
            if random.random() > 0.3:  # 30% chance of flipping
                transform_list.append(transforms.RandomVerticalFlip(p=1.0))
                
            rotational_angles = [90, 180, 270]
            if random.random() > 0.3: # 30% chance of rotating
                rotation_angle = random.choice(rotational_angles)
                transform_list.append(transforms.RandomRotation(degrees=(rotation_angle, rotation_angle)))

            transform = transforms.Compose(transform_list)
            transformed_img = transform(img)
            image_set.append(transformed_img)
            
        return image_set
        
    @staticmethod
    def visual(data, n=1): 
        plt.figure(figsize=(12, 6))
        for i in range(n):
            img = data[i]
            
            # Convert from (C,H,W) to (H,W,C) for display
            if img.dim() == 3 and img.size(0) == 3:  # If it's a color image with 3 channels
                img_display = img.permute(1, 2, 0).cpu().numpy()
            else:
                img_display = img.squeeze().cpu().numpy()
                
            plt.subplot(2, n, i + 1)
            plt.imshow(img_display)
            plt.title("Original")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
            


class NoisyCBSD68_dataset:
    def __init__(self, 
                 batch_size=64,
                 noise_std=0.3):
        
        self.batch_size = batch_size
        
        
        self.train_data = NoisyCBSD68(target_count=200, noise_std=noise_std)
        self.test_data = NoisyCBSD68(target_count=40, noise_std=noise_std)
        
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=batch_size, shuffle=True)
        
    def shape(self):    
        return self.train_data[0][0].shape
        
    def visual(self, n=5): 
        plt.figure(figsize=(12, 6))
        for i in range(n):
            noisy_img, img, _ = self.test_data[i]
            
            # Convert from (C,H,W) to (H,W,C) for display
            if img.dim() == 3 and img.size(0) == 3:  # If it's a color image with 3 channels
                img_display = img.permute(1, 2, 0).cpu().numpy()
                noisy_img_display = noisy_img.permute(1, 2, 0).cpu().numpy()
            else:
                img_display = img.squeeze().cpu().numpy()
                noisy_img_display = noisy_img.squeeze().cpu().numpy()
            
            # Display original image
            plt.subplot(2, n, i + 1)
            plt.imshow(img_display)
            plt.title("Original")
            plt.axis('off')

            # Display noisy image
            plt.subplot(2, n, n + i + 1)
            plt.imshow(noisy_img_display)
            plt.title("Noisy")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        
def test_denoise_visual_CBSD(model, test_data, n=3, device='mps'):
    fig, axes = plt.subplots(n, 3, figsize=(15, 5*n))
    
    for i in range(n):
        noisy_img, img, _ = test_data[i]
        
        # Convert from (C,H,W) to (H,W,C) for display
        if img.dim() == 3 and img.size(0) == 3:  # If it's a color image with 3 channels
            img_display = img.permute(1, 2, 0).cpu().numpy()
            noisy_img_display = noisy_img.permute(1, 2, 0).cpu().numpy()
            model_input = noisy_img.unsqueeze(0).to(device)  # Add batch dimension for model
        else:
            img_display = img.squeeze().cpu().numpy()
            noisy_img_display = noisy_img.squeeze().cpu().numpy()
            model_input = noisy_img.unsqueeze(0).unsqueeze(0).to(device)
        
        # Display original clean image
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title("Original Clean", fontsize=22)
        axes[i, 0].axis('off')

        # Display noisy image
        axes[i, 1].imshow(noisy_img_display)
        axes[i, 1].set_title("Noisy", fontsize=22)
        axes[i, 1].axis('off')
        
        # Generate and display denoised image
        model.eval()
        with torch.no_grad():
            denoised_img = model(model_input)
            if img.dim() == 3 and img.size(0) == 3:  # If it's a color image
                denoised_img = denoised_img.squeeze(0).cpu().numpy()
                denoised_img = np.transpose(denoised_img, (1, 2, 0))  # Change from (C,H,W) to (H,W,C)
            else:
                denoised_img = denoised_img.squeeze().cpu().numpy()
            axes[i, 2].imshow(denoised_img)
            axes[i, 2].set_title("Denoised", fontsize=22)
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def example_usage():
    '''Example Usage of NoisyCBSD68_dataset'''
    
    
    noisy_cbsd68 = NoisyCBSD68_dataset(batch_size=64, noise_std=0.2)
    
    print(len(noisy_cbsd68.train_data))
    print(len(noisy_cbsd68.test_data))
    print(len(noisy_cbsd68.test_data[0]))
    print(noisy_cbsd68.train_data[0][0].shape)
    
    print("Shape of image: ", noisy_cbsd68.shape())
    noisy_cbsd68.visual(n=5)
  
# example_usage()
