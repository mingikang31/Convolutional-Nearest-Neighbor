"""Utility Functions for the Project"""
import torch.nn as nn
import os 
from typing import Optional

def print_cuda_info():
    import torch 

    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Get number of GPUs
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    # Get current GPU name
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
        # Get memory info (in bytes)
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Allocated memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Cached memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

def write_to_file(file_path, data):
    with open(file_path, 'w') as file:
        if isinstance(data, list):
            # For lists like train_eval_results
            for item in data:
                file.write(f"{item}\n")
        elif hasattr(data, '__dict__'):
            # For objects like args
            for key, value in vars(data).items():
                file.write(f"{key}: {value}\n")
        elif isinstance(data, nn.Module):
            # For PyTorch models
            file.write(str(data))
        else:
            # Default case
            file.write(str(data))
            file.write("\n")

def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def argsSimpleNameSpace():
    from types import SimpleNamespace
        
    args = SimpleNamespace(
        a = 1, 
        b = 2, 
        c = 3
    )
