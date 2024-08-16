# Copyright (c) Mingi Kang | mkang817415. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import random
import matplotlib.pyplot as plt

import scipy
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

from mnist1d.data import make_dataset


#############2D Datasets###############

'''
   Data for training 2D CNN Models
   - MNIST data 
   - Fashion MNIST data
   - CIFAR10 data
'''
import torch 
import torch.nn as nn
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt 


### Classification Data - MNIST, FashionMNIST, CIFAR10 ###
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
      

class FashionMNIST:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        
        self.train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        self.test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False)

    def shape(self):
        return self.train_data[0][0].shape

    def visual(self):
        plt.figure(figsize=(6, 3)) 
        plt.imshow(self.train_data[0][0].squeeze(), cmap='gray')
        plt.show()
        

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
   
   
### Denoising Data - MNIST, FashionMNIST, CIFAR10### 

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

class NoisyFashionMNIST(datasets.FashionMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, noise_std=0.1):
        super(NoisyFashionMNIST, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.noise_std = noise_std

    def __getitem__(self, index):
        img, target = super(NoisyFashionMNIST, self).__getitem__(index)
        noisy_img = img + self.noise_std * torch.randn_like(img)
        return noisy_img, img, target

class FashionMNIST_denoise:
    def __init__(self, batch_size=64, noise_std=0.3):
        self.batch_size = batch_size
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.train_data = NoisyFashionMNIST(root='./data', train=True, download=True, transform=transform, noise_std=noise_std)
        self.test_data = NoisyFashionMNIST(root='./data', train=False, download=True, transform=transform, noise_std=noise_std)
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False)
        
    def shape(self):
        return self.train_data[0][0].shape
    
    def visual(self):
        noisy_img, img, _ = self.test_data[0]
        img = img.squeeze()
        noisy_img = noisy_img.squeeze()
        
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(img, cmap='gray')
        
        plt.subplot(1, 2, 2)
        plt.title('Noisy Image')
        plt.imshow(noisy_img, cmap='gray')
        
        plt.show()

class NoisyCIFAR10(datasets.CIFAR10):
   def __init__(self, root, train=True, transform=None, target_transform=None, download=False, noise_std=0.1):
      super(NoisyCIFAR10, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
      self.noise_std = noise_std

   def __getitem__(self, index):
      img, target = super(NoisyCIFAR10, self).__getitem__(index)
      noisy_img = img + self.noise_std * torch.randn_like(img)
      return noisy_img, img, target

class CIFAR10_denoise:
    def __init__(self, batch_size=64, noise_std=0.3):
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
      
   
#############1D Datasets###############
'''
   Data for training 1D CNN Models
   - MNIST1D data 
   - https://github.com/greydanus/mnist1d
'''
class MNIST1D():
   
   def __init__(self, seed = None): 

      
      self.data_args = self.get_dataset_args(as_dict=False)

      self.data_args_dict = self.get_dataset_args(as_dict=True)
      
      self.model_args = self.get_model_args(as_dict=False)
      
      self.model_args_dict = self.get_model_args(as_dict=True)
      
      if not seed: 
         self.set_seed(self.data_args.seed)
      else: 
         self.set_seed(seed)
   
   def make_dataset(self): 
      data = make_dataset(self.data_args)
      # Creating dataset of size [Batch, channels, tokens]
      data['x'] = torch.Tensor(data['x']).unsqueeze(1)
      data['x_test'] = torch.Tensor(data['x_test']).unsqueeze(1)
      return data
   
   @staticmethod
   def set_seed(seed):
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      
   def get_dataset_args(self, as_dict=False):
      arg_dict = {'num_samples': 5000,
               'train_split': 0.8,
               'template_len': 12,
               'padding': [36,60],
               'scale_coeff': .4, 
               'max_translation': 48,
               'corr_noise_scale': 0.25,
               'iid_noise_scale': 2e-2,
               'shear_scale': 0.75,
               'shuffle_seq': False,
               'final_seq_length': 40,
               'seed': 42}
      return arg_dict if as_dict else self.ObjectView(arg_dict)
   
   def get_model_args(self, as_dict=False):
      arg_dict = {'input_size': 40,
               'output_size': 10,
               'hidden_size': 256,
               'learning_rate': 1e-2,
               'weight_decay': 0,
               'batch_size': 100,
               'total_steps': 6000,
               'print_every': 1000,
               'eval_every': 250,
               'checkpoint_every': 1000,
               'device': 'mps',
               'seed': 42}
      return arg_dict if as_dict else self.ObjectView(arg_dict)
   
   @staticmethod
   class ObjectView(object):
      def __init__(self, d): self.__dict__ = d
      
class MNIST1D_Plot(): 
   def __init__(self, data=None, data_args=None):
      self.data = data
      self.data_args = data_args
      
   
   '''Functions for Transformation'''
   def pad(self, x, padding): 
      low, high = padding
      p = low + int(np.random.rand()*(high-low+1))
      return np.concatenate([x, np.zeros((p))])

   def shear(self, x, scale=10):
      coeff = scale*(np.random.rand() - 0.5)
      return x - coeff*np.linspace(-0.5,.5,len(x))

   def translate(self, x, max_translation):
      k = np.random.choice(max_translation)
      return np.concatenate([x[-k:], x[:-k]])

   def corr_noise_like(self, x, scale):
      noise = scale * np.random.randn(*x.shape)
      return gaussian_filter(noise, 2)

   def iid_noise_like(self, x, scale):
      noise = scale * np.random.randn(*x.shape)
      return noise

   def interpolate(self, x, N):
      scale = np.linspace(0,1,len(x))
      new_scale = np.linspace(0,1,N)
      new_x = interp1d(scale, x, axis=0, kind='linear')(new_scale)
      return new_x

   def transform(self, x, y, args, eps=1e-8):
      new_x = self.pad(x+eps, args.padding) # pad
      new_x = self.interpolate(new_x, args.template_len + args.padding[-1])  # dilate
      new_y = self.interpolate(y, args.template_len + args.padding[-1])
      new_x *= (1 + args.scale_coeff*(np.random.rand() - 0.5))  # scale
      new_x = self.translate(new_x, args.max_translation)  #translate
      
      # add noise
      mask = new_x != 0
      new_x = mask*new_x + (1-mask)*self.corr_noise_like(new_x, args.corr_noise_scale)
      new_x = new_x + self.iid_noise_like(new_x, args.iid_noise_scale)
      
      # shear and interpolate
      new_x = self.shear(new_x, args.shear_scale)
      new_x = self.interpolate(new_x, args.final_seq_length) # subsample
      new_y = self.interpolate(new_y, args.final_seq_length)
      return new_x, new_y
   
   
   '''Additional Functions for plotting'''
   def apply_ablations(self, arg_dict, n=7): 
      ablations = [('shear_scale', 0),
                  ('iid_noise_scale', 0),
                  ('corr_noise_scale', 0),
                   ('max_translation', 1),
                   ('scale_coeff', 0),
                   ('padding', [arg_dict['padding'][-1], arg_dict['padding'][-1]]),
                   ('padding', [0, 0]),]
      num_ablations = min(n, len(ablations))
      for i in range(num_ablations):
          k, v = ablations[i]
          arg_dict[k] = v
      return arg_dict
   
   def get_templates(self):
      d0 = np.asarray([5,6,6.5,6.75,7,7,7,7,6.75,6.5,6,5])
      d1 = np.asarray([5,3,3,3.4,3.8,4.2,4.6,5,5.4,5.8,5,5])
      d2 = np.asarray([5,6,6.5,6.5,6,5.25,4.75,4,3.5,3.5,4,5])
      d3 = np.asarray([5,6,6.5,6.5,6,5,5,6,6.5,6.5,6,5])
      d4 = np.asarray([5,4.4,3.8,3.2,2.6,2.6,5,5,5,5,5,5])
      d5 = np.asarray([5,3,3,3,3,5,6,6.5,6.5,6,4.5,5])
      d6 = np.asarray([5,4,3.5,3.25,3,3,3,3,3.25,3.5,4,5])
      d7 = np.asarray([5,7,7,6.6,6.2,5.8,5.4,5,4.6,4.2,5,5])
      d8 = np.asarray([5,4,3.5,3.5,4,5,5,4,3.5,3.5,4,5])
      d9 = np.asarray([5,4,3.5,3.5,4,5,5,5,5,4.7,4.3,5])

      x = np.stack([d0,d1,d2,d3,d4,d5,d6,d7,d8,d9])
      x -= x.mean(1,keepdims=True) # whiten
      x /= x.std(1,keepdims=True)
      x -= x[:,:1]  # signal starts and ends at 0

      templates = {'x': x/6., 't': np.linspace(-5, 5, len(d0))/6.,
               'y': np.asarray([0,1,2,3,4,5,6,7,8,9])}
      return templates
   
   @staticmethod
   class ObjectView(object):
      def __init__(self, d): self.__dict__ = d
 
   '''Plotting Functions'''

   # Main plotting function for MNIST1D data -> I do not think this necessarily has to be in the class
   # Can be an individual function
   def plot_signals(self, xs, t, labels=None, args=None, title=None, ratio=2.6, do_transform=False, dark_mode=False, zoom=1):
      

      rows, cols = 1, 10
      fig = plt.figure(figsize=[cols*1.5,rows*1.5*ratio], dpi=60)
      for r in range(rows):
         for c in range(cols):
            ix = r*cols + c
            x, t = xs[ix], t
            
            # Ensure x is a 1D array if it's a 2D array with a single row
            if x.ndim > 1 and x.shape[0] == 1:
                x = x.squeeze(0)
            ax = plt.subplot(rows,cols,ix+1)

            # plot the data
            if do_transform:
                  assert args is not None, "Need an args object in order to do transforms"
                  x, t = self.transform(x, t, args)  # optionally, transform the signal in some manner
            if dark_mode:
                  plt.plot(x, t, 'wo', linewidth=6)
                  ax.set_facecolor('k')
            else:
                  plt.plot(x, t, 'k-', linewidth=2)
            if labels is not None:
                  plt.title("label=" + str(labels[ix]), fontsize=22)
            plt.xlim(-zoom,zoom) ; plt.ylim(-zoom,zoom)
            plt.gca().invert_yaxis() ; plt.xticks([], []), plt.yticks([], [])
      if title is None:
         fig.suptitle('Noise free', fontsize=24, y=1.1)
      else:
         fig.suptitle(title, fontsize=24, y=1.1)
      plt.subplots_adjust(wspace=0, hspace=0)
      plt.tight_layout() ; plt.show()
      
