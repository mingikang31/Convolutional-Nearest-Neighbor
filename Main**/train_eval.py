'''Training & Evaluation Module for Convolutional Neural Networks'''

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import time 

from utils import set_seed



def Train_Eval(args, 
               model: nn.Module, 
               train_loader, 
               test_loader
               ):
    
    if args.seed != 0:
        set_seed(args.seed)
    
    if args.criterion == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'MSE':
        criterion = nn.MSELoss()

    
    # Optimizer 
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        

    # Learning Rate Scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

        
    # Device
    device = args.device
    model.to(device)
    criterion.to(device)
    
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    
    # Training Loop
    epoch_times = [] # Average Epoch Time 
    epoch_results = [] 
    
    max_accuracy = 0.0 
    max_epoch = 0
    
    for epoch in range(args.num_epochs):
    # for epoch in tqdm(range(args.num_epochs), desc=f"Epoch {epoch+1}/{args.num_epochs}"):
        # Model Training
        model.train() 
        running_loss = 0.0 
        epoch_result = ""
        
        start_time = time.time()
        
        for images, labels in train_loader: 
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # use mixed precision training
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                if hasattr(args, 'clip_grad_norm') and args.clip_grad_norm is not None:
                    scaler.unscale_(optimizer) # Unscale gradients before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:    
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                if hasattr(args, 'clip_grad_norm') and args.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
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
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                else: 
                    outputs = model(images)
                top1, top5 = accuracy(outputs, labels, topk=(1, 5))
                top1_5[0] += top1.item()
                top1_5[1] += top5.item()
        
        top1_5[0] /= len(test_loader)
        top1_5[1] /= len(test_loader)
        epoch_result += f"Accuracy: Top1: {top1_5[0]:.4f}%, Top5: {top1_5[1]:.4f}%"
        print(epoch_result)
        epoch_results.append(epoch_result)
        
        # Max Accuracy Check
        if top1_5[0] > max_accuracy:
            max_accuracy = top1_5[0]
            max_epoch = epoch + 1    
            
        # Learning Rate Scheduler Step
        if scheduler: 
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(top1_5[0])
            else:
                scheduler.step()
                
                
    epoch_results.append(f"\nAverage Epoch Time: {sum(epoch_times) / len(epoch_times):.4f}s")
    epoch_results.append(f"Max Accuracy: {max_accuracy:.4f}% at Epoch {max_epoch}")

    
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