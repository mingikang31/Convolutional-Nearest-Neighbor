'''ImageNet, TinyImageNet, CIFAR100, CIFAR10, MNIST Datasets'''
import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from PIL import Image

class AddGaussianNoise(object):
            def __init__(self, mean=0., std=0.1):
                self.mean = mean
                self.std = std
                
            def __call__(self, tensor):
                return tensor + torch.randn(tensor.size()) * self.std + self.mean


class ImageNet(datasets.ImageNet):
    def __init__(self, args):
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_data = datasets.ImageNet(root=args.data_path, split='train', download=False, transform=transform)
        self.test_data = datasets.ImageNet(root=args.data_path, split='val', download=False, transform=transform)
        
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

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

# TinyImageNet is not available in torchvision.datasets by default.
# You need to implement a custom TinyImageNet dataset class or use an external implementation.
# The following is a placeholder for TinyImageNet, which raises NotImplementedError if used.

class TinyImageNet:
    def __init__(self, args):
        raise NotImplementedError("TinyImageNet dataset is not available in torchvision.datasets. Please implement a custom loader or use an external implementation.")

### TODO : Data Augmentation for CIFAR100 and CIFAR10
# import torchvision.transforms as T

# # These are the standard mean and std values for CIFAR-100
# CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
# CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# transform_train = T.Compose([
#     # 1. Augmentation: Create variations of the image
#     T.RandomCrop(32, padding=4),
#     T.RandomHorizontalFlip(p=0.5),
    
#     # 2. Conversion: Turn the image into a PyTorch tensor
#     T.ToTensor(),
    
#     # 3. Standardization: Normalize the data
#     T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
# ])


class CIFAR100(datasets.CIFAR100): 
    def __init__(self, args): 

        # Define transforms
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
            AddGaussianNoise(mean=0., std=args.noise)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
        
        if args.resize:
            self.upscale_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
            ])
            self.img_size = (3, 224, 224)
        else:
            self.upscale_transform = None 
            self.img_size = (3, 32, 32)
        
        self.train_data = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=train_transform)
        self.test_data = datasets.CIFAR100(root=args.data_path, train=False, download=False, transform=test_transform)
        
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        self.num_classes = 100
        self.upscale_dataset()

    
    def upscale_dataset(self):  
        if self.upscale_transform:
            upscaled_train = datasets.CIFAR100(root=self.train_data.root, train=True, download=True, transform=self.upscale_transform)
            upscaled_test = datasets.CIFAR100(root=self.test_data.root, train=False, download=False, transform=self.upscale_transform)
            
            # update the train and test data
            self.train_data = upscaled_train
            self.test_data = upscaled_test
            
            # update the train and test loaders
            self.train_loader = DataLoader(dataset=self.train_data, batch_size=self.train_loader.batch_size, shuffle=True, num_workers=4)
            self.test_loader = DataLoader(dataset=self.test_data, batch_size=self.test_loader.batch_size, shuffle=False, num_workers=4)
        else:
            print("Upscale transform not defined. Skipping dataset upscale.")
    
    
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
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
            AddGaussianNoise(mean=0., std=args.noise)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
        
        if args.resize:
            self.upscale_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
            ])
            self.img_size = (3, 224, 224)
        else:
            self.upscale_transform = None 
            self.img_size = (3, 32, 32)
      
      
        
        self.train_data = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        self.test_data = datasets.CIFAR10(root=args.data_path, train=False, download=False, transform=test_transform)
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

        self.num_classes = 10

        self.upscale_dataset()
      
    def upscale_dataset(self):
        if self.upscale_transform:
            upscaled_train = datasets.CIFAR10(root=self.train_data.root, train=True, download=False, transform=self.upscale_transform)
            upscaled_test = datasets.CIFAR10(root=self.test_data.root, train=False, download=False, transform=self.upscale_transform)
            
            # update the train and test data
            self.train_data = upscaled_train
            self.test_data = upscaled_test
            
            # update the train and test loaders
            self.train_loader = DataLoader(dataset=self.train_data, batch_size=self.train_loader.batch_size, shuffle=True, num_workers=4)
            self.test_loader = DataLoader(dataset=self.test_data, batch_size=self.test_loader.batch_size, shuffle=False, num_workers=4)
        else:
            print("Upscale transform not defined. Skipping dataset upscale.")
      
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
