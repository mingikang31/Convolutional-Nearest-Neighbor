import torch 
import torch.nn as nn 
import torchvision.transforms as T 
from torch.optim import AdamW
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
import numpy as np 

class PatchEmbedding(nn.Module): 
    def __init__(self, d_model, img_size, patch_size, n_channels=3): 
        super(PatchEmbedding, self).__init__()
        
        self.d_model = d_model # Dimensionality of Model 
        self.img_size = img_size # Size of Image
        self.patch_size = patch_size # Patch Size 
        self.n_channels = n_channels # Number of Channels in Image
        
        self.linear_projection = nn.Conv2d(in_channels=n_channels, out_channels=d_model, kernel_size=patch_size, stride=patch_size) # Linear Projection Layer
        
        self.flatten = nn.Flatten(start_dim=2)
        
    def forward(self, x): 
        x = self.linear_projection(x) # (B, C, H, W) -> (B, d_model, H', W')
        x = self.flatten(x) # (B, d_model, H', W') -> (B, d_model, n_patches)
        x = x.transpose(1, 2) # (B, d_model, n_patches) -> (B, n_patches, d_model)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length): 
        super(PositionalEncoding, self).__init__()
        
        self.cls_tokens = nn.Parameter(torch.randn(1, 1, d_model)) # Classification Token
        
        # Creating Positional Encoding 
        pe = torch.zeros(max_seq_length, d_model)
        
        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x): 
        # Expand to have class token for each image in batch 
        tokens_batch = self.cls_tokens.expand(x.shape[0], -1, -1) # (B, 1, d_model)
        
        # Concatenate class token with positional encoding
        x = torch.cat((tokens_batch, x), dim=1)
        
        # Add positional encoding to the input 
        x = x + self.pe
        
        return x

class MultiHeadAttention(nn.Module): 
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # dimension of each head
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)        
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output, attn_probs
    
    def split_head(self, x): 
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)
        
    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model) 
    
    def forward(self, x, mask=None):
        q = self.split_head(self.W_q(x)) # (B, num_heads, seq_length, d_k)
        k = self.split_head(self.W_k(x))
        v = self.split_head(self.W_v(x))
        
        attn_output, _ = self.scaled_dot_product_attention(q, k, v, mask) # (B, num_heads, seq_length, d_k)
        output = self.W_o(self.combine_heads(attn_output)) # (B, seq_length, d_model)
        return output
    
class TransformerEncoder(nn.Module): 
    def __init__(self, d_model, num_heads, r_mlp=4):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.r_mlp = r_mlp        
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Multilayer Perceptron 
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp),
            nn.GELU(),
            nn.Linear(d_model * r_mlp, d_model)
        )
        
    def forward(self, x): 
        # Multi-Head Attention
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
        # Feed Forward Network 
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        return x
        

class VisionTransformer(nn.Module): 
    def __init__(self, d_model, img_size, n_classes, n_heads, patch_size, n_channels, n_layers): 
        super(VisionTransformer, self).__init__()
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model # Dimensionality of Model 
        self.img_size = img_size # Image size 
        self.n_classes = n_classes # Number of Classes 
        self.n_heads = n_heads # Number of Heads
        self.patch_size = patch_size # Patch Size
        self.n_channels = n_channels # Number of Channels
        self.n_layers = n_layers # Number of Layers
        
        self.option = 0 # Option for Transformer Encoder
                             # Attention,
                             # Conv2d, 
                             # ConvNN
                             # ConvNN_Attn
                             # Conv2d + ConvNN 
                             # Attention + ConvNN
        
        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
        self.max_seq_length = self.n_patches + 1
        
        
        
        
        # Patch Embedding Layer 
        self.patch_embedding = PatchEmbedding(d_model, self.img_size, self.patch_size, self.n_channels)
        self.positional_encoding = PositionalEncoding(d_model, self.max_seq_length)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(d_model, n_heads) for _ in range(n_layers)]) # Stacking Transformer Encoder Layers
        
        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0]) # Taking the CLS token for classification
        return x       
    
    # after patch embedding:  torch.Size([128, 4, 9])
    # after positional encoding:  torch.Size([128, 5, 9])
    # after transformer encoder:  torch.Size([128, 5, 9])
    # after classifier:  torch.Size([128, 10])
    
    
    
if __name__ == "__main__":
    d_model = 9
    n_classes = 10
    img_size = (32,32)
    patch_size = (16,16)
    n_channels = 3
    n_heads = 3
    n_layers = 3
    batch_size = 128
    epochs = 5
    alpha = 0.005
    
    
    # MNIST Dataset 
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor()
    ])

    train_set = CIFAR10(
        root="./../datasets", train=True, download=True, transform=transform
    )
    test_set = CIFAR10(
        root="./../datasets", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
    
    # Training the Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    transformer = VisionTransformer( 
                    d_model, 
                    img_size, 
                    n_classes, 
                    n_heads, 
                    patch_size, 
                    n_channels, 
                    n_layers
                ).to(device)

    optimizer = AdamW(transformer.parameters(), lr=alpha)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        training_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = transformer(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs} loss: {training_loss  / len(train_loader) :.3f}')
        
    # Testing the Model 
    transformer.eval()
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = transformer(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f} %')
