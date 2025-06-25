'''Training & Evaluation Module for Convolutional Neural Networks'''

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import time 

# Assuming 'utils.py' with set_seed is in the same directory
from utils import set_seed 

import wandb

def accuracy(output, target, topk=(1,)):
    """Computes the top-k accuracy of the model."""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    # Return a list of accuracies for each k
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def Train_Eval(args, 
               model: nn.Module, 
               train_loader, 
               test_loader
               ):
    
    if args.seed != 0:
        set_seed(args.seed)
    
    # -- Criterion --
    if args.criterion == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'MSE':
        criterion = nn.MSELoss()

    # -- Optimizer --
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    # -- Learning Rate Scheduler --
    scheduler = None
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    # -- Device & AMP --
    device = args.device
    model.to(device)
    criterion.to(device)
    
    scaler = None
    if args.use_amp:
        # Enables Automatic Mixed Precision
        scaler = torch.cuda.amp.GradScaler()
    
    # -- Training Loop --
    epoch_times = []
    max_accuracy = 0.0 
    max_epoch = 0
    
    print("--- Starting Training ---")
    for epoch in range(args.num_epochs):
        # --- 1. Model Training Phase ---
        model.train() 
        running_loss_train = 0.0
        top1_5_train = [0, 0] # [top1, top5]
        
        start_time = time.time()
        # Use tqdm for a progress bar
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]"): 
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if args.use_amp and scaler:
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

            # Calculate accuracy for the batch
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            top1_5_train[0] += top1.item()
            top1_5_train[1] += top5.item()
            running_loss_train += loss.item()
            
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # --- 2. Model Evaluation Phase ---
        model.eval()
        running_loss_test = 0.0
        top1_5_test = [0, 0] # [top1, top5]
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Test]"): 
                images, labels = images.to(device), labels.to(device)
                
                # AMP is used for inference as well for consistency
                if args.use_amp and scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                else: 
                    outputs = model(images)
                    
                loss = criterion(outputs, labels)
                running_loss_test += loss.item()
                
                # Calculate accuracy for the batch
                top1, top5 = accuracy(outputs, labels, topk=(1, 5))
                top1_5_test[0] += top1.item()
                top1_5_test[1] += top5.item()

    
        # --- 3. Calculate and Log Metrics for the Epoch ---
        
        # Calculate average metrics
        train_loss_epoch = running_loss_train / len(train_loader)
        train_top1_acc_epoch = top1_5_train[0] / len(train_loader)
        train_top5_acc_epoch = top1_5_train[1] / len(train_loader)

        test_loss_epoch = running_loss_test / len(test_loader)
        test_top1_acc_epoch = top1_5_test[0] / len(test_loader)
        test_top5_acc_epoch = top1_5_test[1] / len(test_loader)
        
        print(f"Epoch {epoch + 1} Summary | Time: {epoch_time:.2f}s")
        print(f"\tTrain -> Loss: {train_loss_epoch:.4f}, Top-1 Acc: {train_top1_acc_epoch:.2f}%, Top-5 Acc: {train_top5_acc_epoch:.2f}%")
        print(f"\tTest  -> Loss: {test_loss_epoch:.4f}, Top-1 Acc: {test_top1_acc_epoch:.2f}%, Top-5 Acc: {test_top5_acc_epoch:.2f}%")

        # Consolidated logging to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss_epoch,
            "train/top1_acc": train_top1_acc_epoch,
            "train/top5_acc": train_top5_acc_epoch,
            "test/loss": test_loss_epoch,
            "test/top1_acc": test_top1_acc_epoch,
            "test/top5_acc": test_top5_acc_epoch,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch_time_s": epoch_time,
        })
        
        
        # --- 4. Post-Epoch Operations ---
        
        # Check if this is the best model so far
        if test_top1_acc_epoch > max_accuracy:
            max_accuracy = test_top1_acc_epoch
            max_epoch = epoch + 1
            torch.save(model.state_dict(), args.output_dir + f'/best_model.pth')
            wandb.save('best_model.pth')

        if epoch % 5 == 0: 
            torch.save(model.state_dict(), args.output_dir + f'/model_epoch_{epoch + 1}.pth')
            wandb.save(f'model_epoch_{epoch + 1}.pth')

        # Step the learning rate scheduler
        if scheduler: 
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_top1_acc_epoch)
            else:
                scheduler.step()
    
    print("--- Training Finished ---")
    print(f"Max Test Accuracy: {max_accuracy:.2f}% achieved at Epoch {max_epoch}")
    return {"max_test_accuracy": max_accuracy, "best_epoch": max_epoch}
