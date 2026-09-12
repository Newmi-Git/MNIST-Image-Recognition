import torch.nn as nn


class ImageRecog(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.network(x)