from model import CNN
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
    ])

train_dataset = datasets.CIFAR10(root = "./data",train=True,download=True,transform=transform)
test_dataset = datasets.CIFAR10(root = "./data" , train=False , download = True , transform=transform)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=True)

model = CNN(32)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    running_loss = 0.00
    for batch_idx,(img,label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/100], Loss: {running_loss/len(train_loader):.4f}')
        
        