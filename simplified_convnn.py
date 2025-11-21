import torch 
import torch.nn as nn 
import torch.nn.functional as F


class ConvNN(nn.Module):
    def __init__(self, in_channels, out_channels, k=9, rho='softmax'):
        super().__init__()
        self.k, self.rho = k, rho
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=k)
        self.w_k = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1)
        self.w_q = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1)
        self.w_v = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1)
       
    def forward(self, x):        
        # Project the input
        k, q, v = self.w_k(x), self.w_q(x), self.w_v(x)

        # Compute Similarities
        k_norm = F.normalize(k, p=2, dim=1)
        q_norm = F.normalize(q, p=2, dim=1)
        S = k_norm.transpose(1, 2) @ q_norm        

        # Select Neighbors
        kmax, kargmax = torch.topk(S, k=self.k, dim=-1)
        if self.rho == 'softmax':
            kmax = torch.softmax(kmax, dim=-1)

        # Neighbor Gathering and Modulation
        v = v.unsqueeze(-1).expand(*x.shape, self.k)
        kmax = kmax.unsqueeze(1).expand(*x.shape, self.k)
        kargmax = kargmax.unsqueeze(1).expand(*x.shape, self.k)
                
        x_nn = kmax * torch.gather(v, dim=2,
                                   index=kargmax)                  
        # Neighbor Aggregation
        b, c, _ = x.shape
        return self.conv1d(x_nn.view(b, c, -1))

if __name__ == "__main__":
    ex = torch.randn(32, 3, 32) # [Batch, Channels, Length]
    model = ConvNN(in_channels=3, out_channels=16, k=9, rho='softmax')
    out = model(ex)
    print(out.shape)