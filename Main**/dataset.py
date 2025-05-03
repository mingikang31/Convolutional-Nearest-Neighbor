'''ImageNet, CIFAR100, CIFAR10, MNIST Datasets'''
import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

class ImageNet(datasets.ImageNet):
    def __init__(self, args):
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_data = datasets.ImageNet(root=args.data_path, split='train', download=getattr(args, 'download', False), transform=transform)
        self.test_data = datasets.ImageNet(root=args.data_path, split='val', download=getattr(args, 'download', False), transform=transform)
        
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False)
        
        self.num_classes = 1000
        self.img_size = (3, 224, 224)
    
    def shape(self):
        return self.train_data[0][0].shape
    
    def visual(self):
        # Get normalized tensor
        img = self.test_data[0][0]
        # Denormalize for visualization
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = img.permute(1, 2, 0).clamp(0, 1)
        plt.figure(figsize=(6, 3))
        plt.imshow(img)
        plt.show()
        
class CIFAR100(datasets.CIFAR100): 
    def __init__(self, args): 
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])
        
        self.train_data = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform)
        self.test_data = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform)
        
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False)
        
        self.num_classes = 100
        self.img_size = (3, 32, 32)
    
    def shape(self):
        return self.train_data[0][0].shape
    
    def visual(self):
        # Get normalized tensor
        img = self.test_data[0][0]
        # Denormalize for visualization
        img = img * torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1) + torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
        img = img.permute(1, 2, 0).clamp(0, 1)
        plt.figure(figsize=(6, 3))
        plt.imshow(img)
        plt.show()
        
class CIFAR10(datasets.CIFAR10): 
   def __init__(self, args):
      # Define transforms
      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
      ])
      
      self.train_data = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
      self.test_data = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
      self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True)
      self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False)
      
      self.num_classes = 10 
      self.img_size = (3, 32, 32)
      
   def shape(self): 
      return self.train_data[0][0].shape
   
   def visual(self): 
      # Get normalized tensor
      img = self.test_data[0][0]
      # Denormalize for visualization
      img = img * torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
      img = img.permute(1, 2, 0).clamp(0, 1)
      plt.figure(figsize=(6, 3)) 
      plt.imshow(img)
      plt.show()      

class MNIST(datasets.MNIST): 
    def __init__(self, args):
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        
        self.train_data = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform)
        self.test_data = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False)
        
        self.num_classes = 10
        self.img_size = (1, 28, 28)
        
    def shape(self):
        return self.train_data[0][0].shape
    
    def visual(self):
        # Get normalized tensor
        img = self.test_data[0][0]
        # Denormalize for visualization
        img = img * torch.tensor([0.3081]).view(1, 1, 1) + torch.tensor([0.1307]).view(1, 1, 1)
        img = img.permute(1, 2, 0).squeeze().clamp(0, 1)
        plt.figure(figsize=(6, 3))
        plt.imshow(img, cmap='gray')
        plt.show()