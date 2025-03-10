'''Denoising Models for HPC'''

import torch
import torch.nn as nn



'''
Autoencoder Architecture for Denoising
'''

class Denoising_AutoEncoder_CNN(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1,
                 kernel_size=3,
                 stride=1,
                 padding=1
            ):
        super(Denoising_AutoEncoder_CNN, self).__init__() 
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.name = "Denoising_AutoEncoder_CNN"
        
        
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
    
