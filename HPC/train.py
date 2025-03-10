'''Training and Testing Functions for 2D models'''

import torch 
import torch.nn 
import torch.optim as optim 

import time





def train_classification(model, 
                         device,
                         training_type,
                         train_loader,
                         criterion=None, 
                         optimizer=None, 
                         lr = 1e-3,
                         num_epochs=50,
                         output_file='output.txt'):
                         
    if criterion is None: 
        criterion = torch.nn.CrossEntropyLoss()
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    epoch_times = []
    for epoch in range(num_epochs):
        start = time.time()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to("mps"), labels.to("mps")
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        end = time.time()
        epoch_times.append(end - start)
        print(f'Epoch {epoch+1}, Time: {end - start}, Loss: {running_loss/len(train_loader)}', file=output_file)
    
    
    
    print(f'Training Completed for 
          Average epoch time: {sum(epoch_times)/num_epochs}', file=output_file)
    print(f"Total ")
    
def test_classification(model, 
                        device,
                        test_loader, 
                        output_file='output'):
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
    print(f'Accuracy on test set: {accuracy}%', file=output_file)
    return accuracy

    
