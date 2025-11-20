'''ImageNet, TinyImageNet, CIFAR100, CIFAR10, MNIST Datasets'''
import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from PIL import Image

class CIFAR100(datasets.CIFAR100): 
    def __init__(self, args): 

        self.CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        self.CIFAR100_STD = (0.2675, 0.2565, 0.2761)

        # Train Transformations
        self.train_transform_list = []
        if args.resize: 
            self.train_transform_list += [transforms.Resize((args.resize, args.resize))]
        self.train_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR100_MEAN, std=self.CIFAR100_STD),
            ]
        
        # Test Transformations
        self.test_transform_list = []
        if args.resize: 
            self.test_transform_list += [transforms.Resize((args.resize, args.resize))]
        self.test_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR100_MEAN, std=self.CIFAR100_STD)
        ]
        
        # Define transforms
        train_transform = transforms.Compose(self.train_transform_list)
        test_transform = transforms.Compose(self.test_transform_list)

        # Load Datasets
        self.train_data = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=train_transform)
        self.test_data = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=test_transform)

        # Data Loaders
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Set image size and number of classes
        self.img_size = (3, args.resize, args.resize) if args.resize else (3, 32, 32)
        self.num_classes = 100

    def shape(self):
        return self.train_data[0][0].shape
    
    def visual(self): 
        img = self.test_data[0][0]
        # Use self.CIFAR100_STD and self.CIFAR100_MEAN instead of hardcoded values
        img = img * torch.tensor(self.CIFAR100_STD).view(3, 1, 1) + torch.tensor(self.CIFAR100_MEAN).view(3, 1, 1)
        img = img.permute(1, 2, 0).clamp(0, 1)
        plt.figure(figsize=(6, 3)) 
        plt.imshow(img)
        plt.axis('off')  # Optional: cleaner look
        plt.show()
        
class CIFAR10(datasets.CIFAR10): 
    def __init__(self, args):

        self.CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        self.CIFAR10_STD = (0.2470, 0.2435, 0.2616)

        # Transformations
        # Train Transformations
        self.train_transform_list = []
        if args.resize: 
            self.train_transform_list += [transforms.Resize((args.resize, args.resize))]
        self.train_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR10_MEAN, std=self.CIFAR10_STD),
            ]

        # Test Transformations
        self.test_transform_list = []
        if args.resize: 
            self.test_transform_list += [transforms.Resize((args.resize, args.resize))]
        self.test_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR10_MEAN, std=self.CIFAR10_STD)
        ]
        
        # Define transforms
        train_transform = transforms.Compose(self.train_transform_list)
        test_transform = transforms.Compose(self.test_transform_list)

        # Load Datasets
        self.train_data = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        self.test_data = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)

        # Data Loaders
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Set image size and number of classes
        self.img_size = (3, args.resize, args.resize) if args.resize else (3, 32, 32)
        self.num_classes = 10

    def shape(self): 
        return self.train_data[0][0].shape
    
    def visual(self): 
        img = self.test_data[0][0]
        # Use self.CIFAR10_STD and self.CIFAR10_MEAN instead of hardcoded values
        img = img * torch.tensor(self.CIFAR10_STD).view(3, 1, 1) + torch.tensor(self.CIFAR10_MEAN).view(3, 1, 1)
        img = img.permute(1, 2, 0).clamp(0, 1)
        plt.figure(figsize=(6, 3)) 
        plt.imshow(img)
        plt.axis('off')
        plt.show()