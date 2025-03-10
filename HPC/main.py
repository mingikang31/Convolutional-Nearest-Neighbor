'''Main script to run experiments on HPC'''

import torch
import torch.nn as nn

import os
import sys
import time 

if torch.cuda.is_available():
    print(f"Device Available: {torch.cuda.get_device_name(0)} \n")
    device = torch.device('cuda')
else: 
    device = torch.device('cpu')
    print("Device Available: CPU \n")
    








prefixes = ['Classification', 'Denoising']

data_types = ['CIFAR10', 'BSD68']


def run_experiment(prefix, data_type):
    print(f'Running experiment for {prefix} on {data_type}')
    
    
    
def __main__():
    for prefix in prefixes:
        for data_type in data_types:
            run_experiment(prefix, data_type)
            
    
if __name__ == '__main__':
    __main__()
