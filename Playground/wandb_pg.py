# Torch
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim 


# Train + Data 
import sys 
sys.path.append('../Layers')
from Conv1d_NN_spatial import * 
from Conv2d_NN_spatial import * 

sys.path.append('../Data')
from CIFAR10 import * 


sys.path.append('../Models')
from models_2d import *

sys.path.append('../Train')
from train2d import * 


import time

# Import the W&B Python Library
import wandb


cifar10 = CIFAR10()
train_loader = cifar10.train_loader
test_loader = cifar10.test_loader

model = ConvNN_2D_K_All()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 1. Start a W&B Run
run = wandb.init(project="ConvNN - Convolutional Nearest Neighbor", notes="", tags=["ConvNN_2D_K_All"], name="ConvNN_2D_K_All")

wandb.config = {"epochs": 10, "learning_rate": 0.001, "batch_size": 64}


# Training Model
epoch_times = []
for epoch in range(wandb.config['epochs']):
    model.train()
    start = time.time()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to('mps'), labels.to('mps') ## TODO edit later
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    end = time.time()
    print(f'Epoch {epoch+1}, Time: {end - start}, Loss: {running_loss/len(train_loader)}')
    epoch_times.append( end - start )
    
    
    # Testing Model 
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('mps'), labels.to('mps') ## TODO edit later
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
    
    wandb.log({"epoch": epoch, "loss": running_loss/len(train_loader), "epoch_time": end - start, "accuracy": accuracy})

    
print(f'\n Average epoch time: {sum(epoch_times)/len(epoch_times)}')


# 4. Log an artifact to W&B
# wandb.log_artifact(model)


run.finish()

# Optional: save model at the end
# model.to_onnx()
# wandb.save("model.onnx")
