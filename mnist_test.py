import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

class ImageRecog(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(786, 128),
            nn.ReLU,
            
            nn.Linear(128, 64),
            nn.ReLU,
            
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.network(x)
    
model = ImageRecog()


    
    

dataset = datasets.MNIST(
    root="datasets",
    train=True,
    download=True,
    transform=ToTensor()
)

train_loader = DataLoader(
    dataset,
    batch_size = 64,
    shuffle=True
)

image, label = dataset[0]

print("Image Shape: ", image.shape)
flattened = image.flatten()



plt.imshow(image.squeeze(),cmap="gray")
plt.title(f"Digit: {label}")
plt.axis("off")
plt.show()




