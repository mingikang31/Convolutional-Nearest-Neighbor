'''Training & Evaluation Module for Convolutional Neural Networks'''

import torch
import torch.nn as nn
import torch.optim as optim

import time 

def Train_Eval(args, 
               model: nn.Module, 
               train_loader, 
               test_loader
               ):
    
    # Criterion 
    if args.criterion == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'MSE':
        criterion = nn.MSELoss()
    
    # Optimizer 
    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Device
    device = args.device
    model.to(device)
    criterion.to(device)
    
    # Training Loop
    epoch_times = [] # Average Epoch Time 
    epoch_results = [] 
    for epoch in range(args.num_epochs):
        # Model Training
        model.train() 
        running_loss = 0.0 
        epoch_result = ""
        
        start_time = time.time()
        
        for images, labels in train_loader: 
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        end_time = time.time()
        epoch_result += f"[Epoch {epoch+1}] Time: {end_time - start_time:.4f}s, Loss: {running_loss/len(train_loader):.8f} | "
        epoch_times.append(end_time - start_time)
        
        # Model Evaluation 
        model.eval()
        top1_5 = [0, 0]
        with torch.no_grad():
            for images, labels in test_loader: 
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                top1, top5 = accuracy(outputs, labels, topk=(1, 5))
                top1_5[0] += top1.item()
                top1_5[1] += top5.item()
        
        top1_5[0] /= len(test_loader)
        top1_5[1] /= len(test_loader)
        epoch_result += f"Accuracy: Top1: {top1_5[0]:.4f}%, Top5: {top1_5[1]:.4f}%"
        print(epoch_result)
        epoch_results.append(epoch_result)
    
    epoch_results.append(f"\nAverage Epoch Time: {sum(epoch_times) / len(epoch_times):.4f}s")
    return epoch_results
        

def accuracy(output, target, topk=(1,)):
    """Computes the top-1 and top-5 accuracy of the model."""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
# [72.5, 91.3] - [top1, top5]