import torch 
import torch.nn as nn 
from torchsummary import summary 
from NNT import NN
from Conv1d_NN import Conv1d_NN

## Configure the device (cuda, mps, cpu)
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)


## Modified Neural Network with Conv1d
model = nn.Sequential(
    Conv1d_NN(in_channels=1, out_channels =32,kernel_size=40, K = 3),  # Change the number of input channels to 1
    nn.ReLU(), 
    nn.MaxPool1d(kernel_size=2, stride=2),
    Conv1d_NN(in_channels=32, out_channels=64, kernel_size=27, K = 3),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=2, stride=2),
    nn.Flatten(), 
    nn.Linear(128, 64),  # Change the number of input features to match the output of the last convolutional layer
    nn.ReLU(), 
    nn.Linear(64, 10) 
).to(device)

# Model Summary 
from torchsummary import summary
# summary(model, (1, 40)) # need cpu to get summary -> Why?


## MNIST 1D Dataset
from mnist1d.data import make_dataset, get_dataset_args

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d
    
def get_dataset_args(as_dict=False):
    arg_dict = {'num_samples': 5000,
            'train_split': 0.8,
            'template_len': 12,
            'padding': [36,60],
            'scale_coeff': .4, 
            'max_translation': 48,
            'corr_noise_scale': 0.25,
            'iid_noise_scale': 2e-2,
            'shear_scale': 0.75,
            'shuffle_seq': False,
            'final_seq_length': 40,
            'seed': 42}
    return arg_dict if as_dict else ObjectView(arg_dict)

# Creating Dataset 
defaults = get_dataset_args()

data = make_dataset(defaults) 
x, y, t = torch.tensor(data['x'], dtype=torch.long).unsqueeze(1), torch.tensor(data['y'], dtype=torch.long), torch.tensor(data['t'], dtype=torch.long)
x_test, y_test = torch.tensor(data['x_test'], dtype=torch.long).unsqueeze(1), torch.tensor(data['y_test'], dtype=torch.long)


## Configure data loader
from torch.utils.data import Dataset, DataLoader 

class MNIST1DDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = MNIST1DDataset(x, y) 
train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = MNIST1DDataset(x_test, y_test)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False)

## Train the Model 
# Configure Training
from torch.optim import Adam 
loss_fn = nn.CrossEntropyLoss()
opt = Adam(model.parameters(), lr=1e-3)

def train_batch(x, y, model, opt, loss_fn): 
    model.train() 
    
    opt.zero_grad()
    batch_loss = loss_fn(model(x), y)
    batch_loss.backward()
    opt.step()
    
    return batch_loss.detach().cpu().numpy()

@torch.no_grad()
def accuracy(x, y, model): 
    model.eval() 
    
    prediction = model(x)
    argmaxes = prediction.argmax(dim=1)
    s = torch.sum(((argmaxes == y).float())/len(y))
    return s.cpu().numpy()

# Training Network 
import numpy as np

losses, accuracies = [], [] 
n_epochs = 5

for epoch in range(n_epochs):
    print(f"Running epoch {epoch+1} of {n_epochs}")
    
    epoch_losses, epoch_accuracies = [], []
    
    for batch in train_dl: 
        x, y = batch 
        batch_loss = train_batch(x, y, model, opt, loss_fn)
        epoch_losses.append(batch_loss)
    epoch_loss = np.mean(epoch_losses) 
    
    for batch in train_dl: 
        x, y = batch 
        batch_acc = accuracy(x, y, model)
        epoch_accuracies.append(batch_acc)
    epoch_accuracy = np.mean(epoch_accuracies)
    
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)
        
# Visualize how it did during training 
import matplotlib.pyplot as plt
plt.figure(figsize=(13, 3))
plt.subplot(121)
plt.title("Training Loss value over epochs")
plt.plot(np.arange(n_epochs) + 1, losses)
plt.subplot(122)
plt.title("Testing Accuracy value over epochs") 
plt.plot(np.arange(n_epochs) + 1, accuracies)

# Testing the learned classifier 
epoch_accuracies = [] 
for ix, batch in enumerate(iter(test_dl)): 
    x, y = batch 
    batch_acc = accuracy(x, y, model)
    epoch_accuracies.append(batch_acc)

print(f"Test accuracy: {np.mean(epoch_accuracies):.6f}")