'''ImageNet, CIFAR100, CIFAR10, MNIST Datasets'''

from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 


## 
class ImageNet(datasets.ImageNet):
    def __init__(self, args):
        self.train_data = datasets.ImageNet(root=args.data_path, split='train', download=args.download)
        self.test_data = datasets.ImageNet(root=args.data_path, split='val', download=args.download)
        
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False)
        
        self.num_classes = 1000
        self.img_size = (3, 224, 224)
    
    def shape(self):
        return self.train_data[0][0].shape
    
    def visual(self):
        img = self.test_data[0][0].permute(1, 2, 0)
        plt.figure(figsize=(6, 3))
        plt.imshow(img)
        plt.show()
        
class CIFAR100(datasets.CIFAR100): 
    def __init__(self, args): 
        
        self.train_data = datasets.CIFAR100(root=args.data_path, train=True, download=True)
        self.test_data = datasets.CIFAR100(root=args.data_path, train=False, download=True)
        
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False)
        
        self.num_classes = 100
        self. img_size = (3, 32, 32)
    
    def shape(self):
        return self.train_data[0][0].shape
    
    def visual(self):
        img = self.test_data[0][0].permute(1, 2, 0)
        plt.figure(figsize=(6, 3))
        plt.imshow(img)
        plt.show()
        
class CIFAR10(datasets.CIFAR10): 
   def __init__(self, args):
      
      self.train_data = datasets.CIFAR10(root=args.data_path, train=True, download=True)
      self.test_data = datasets.CIFAR10(root=args.data_path, train=False, download=True)
      self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True)
      self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False)
      
      self.num_classes = 10 
      self.img_size = (3, 32, 32)
      
   def shape(self): 
      return self.train_data[0][0].shape
   
   def visual(self): 
      img = self.test_data[0][0].permute(1, 2, 0)  
      plt.figure(figsize=(6, 3)) 
      plt.imshow(img)
      plt.show()      

class MNIST(datasets.MNIST): 
    def __init__(self, args):
        self.train_data = datasets.MNIST(root=args.data_path, train=True, download=True)
        self.test_data = datasets.MNIST(root=args.data_path, train=False, download=True)
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False)
        
        self.num_classes = 10
        self.img_size = (1, 28, 28)
        
    def shape(self):
        return self.train_data[0][0].shape
    
    def visual(self):
        img = self.test_data[0][0].permute(1, 2, 0)
        plt.figure(figsize=(6, 3))
        plt.imshow(img)
        plt.show()