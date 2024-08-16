'''Training functions for 2D CNN models'''

import time 
import torch
import torch.nn as nn
import torch.optim as optim

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    epoch_times = []
    for epoch in range(num_epochs):
        start = time.time()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to('mps'), labels.to('mps')
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        end = time.time()
        print(f'Epoch {epoch+1}, Time: {end - start}, Loss: {running_loss/len(train_loader)}')
        epoch_times.append( end - start )
    print(f'\n Average epoch time: {sum(epoch_times)/len(epoch_times)}')

# Accuracy evaluation function
def evaluate_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('mps'), labels.to('mps')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy}%')
    return accuracy

# Denoising training function
def train_denoising_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    
    epoch_times = []
    for epoch in range(num_epochs):
        start = time.time()
        running_loss = 0.0
        for noisy_images, clean_images, _ in train_loader:
            noisy_images, clean_images = noisy_images.to('mps'), clean_images.to('mps')
            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, clean_images)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        end = time.time()
        print(f'Epoch {epoch+1}, Time: {end - start}, Loss: {running_loss/len(train_loader)}')
        epoch_times.append( end - start )
        
    print(f'\n Average epoch time: {sum(epoch_times)/len(epoch_times)}')

# Denoising evaluation function
def evaluate_denoising_accuracy(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for noisy_images, clean_images, _ in test_loader:
            noisy_images, clean_images = noisy_images.to('mps'), clean_images.to('mps')
            outputs = model(noisy_images)
            loss = criterion(outputs, clean_images)
            total_loss += loss.item()
    average_loss = total_loss / len(test_loader)
    print(f'Average loss on test set: {average_loss}')
    return average_loss



def evaluate_accuracy_psnr(model, test_loader, criterion):
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for noisy_images, clean_images, _ in test_loader:
            noisy_images, clean_images = noisy_images.to('mps'), clean_images.to('mps')
            outputs = model(noisy_images)
            mse = criterion(outputs, clean_images)
            psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)
            total_psnr += psnr.item()
    average_psnr = total_psnr / len(test_loader)
    print(f'Average PSNR on test set: {average_psnr}')
    return average_psnr


### Example Usage ###   

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3) 
# cnn = cnn.to('mps')

# train_model(cnn, train_loader, criterion, optimizer, num_epochs=10)

# evaluate_accuracy(cnn, test_loader)


'''
Classification : Criterion = nn.CrossEntropyLoss()
Denoising : Criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

'''
