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

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
    
dataset = datasets.MNIST(
    root="datasets",
    train=True,
    download=True,
    transform=ToTensor()
)

test_dataset = datasets.MNIST(
    root="datasets",
    train=False,
    download=True,
    transform=ToTensor()    
)

train_loader = DataLoader(
    dataset,
    batch_size = 64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


images, labels = next(iter(train_loader))


for epoch in range(5):
    total_loss = 0
    for images, labels in train_loader:
        images = images.view(images.size(0), -1)
        prediction = model(images)
        
        loss = loss_function(prediction, labels) #how
        optimizer.zero_grad() #Which
        loss.backward() #How

        optimizer.step() #Do
        
        total_loss += loss.item()
        
    average_loss = total_loss / len(train_loader)
    
    print(f"Epoch: {epoch + 1}/5 Total loss: {total_loss}, Average loss: {average_loss}")

torch.save(model.state_dict(), 'model_weights.pth')

correct = 0
total = 0

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(images.size(0), -1)
        prediction = model(images)
        
        predicted_digits=prediction.argmax(dim=1)
        
        correct += (predicted_digits == labels).sum().item()
        total += labels.size(0)
        
    accuracy = correct / total * 100
    print(f"Test accuracy: ", accuracy)