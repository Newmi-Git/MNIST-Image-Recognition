import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model import ImageRecog


    
model = ImageRecog()

state_dict = torch.load('model_weights.pth')

model.load_state_dict(state_dict)

model.eval()

test_dataset = datasets.MNIST(
    root="datasets",
    train=False,
    download=True,
    transform=ToTensor()    
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

images, labels = next(iter(test_loader))

original_images = images

images = images.view(images.size(0), -1)

with torch.no_grad():
    predictions = model(images)
    
predicted_digits = predictions.argmax(dim=-1)
'''
for i in range(10):
    
    plt.figure()
    
    plt.imshow(original_images[i].squeeze(), cmap="gray")
    
    actual = labels[i].item()
    predicted = predicted_digits[i].item()
    
    plt.title(f"Actual: {actual}\n Predicted: {predicted}")
    plt.show()

'''
