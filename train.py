from model import CNN
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch
    
def train(model,train_loader,criterion,optimizer,device):
    model.train()
    running_loss = 0.00
    correct_predictions = 0
    total_samples = 0
    
    for images,labels in train_loader:
        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _,predicted = torch.max(outputs,1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    
    avg_loss = running_loss/len(train_loader)
    accuracy = (correct_predictions/total_samples)*100
    
    return avg_loss,accuracy

def test(model, test_loader, criterion, device):
    model.eval()  # Modeli test moduna alıyoruz
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Gradyanları devre dışı bırakıyoruz
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Veriyi GPU'ya taşıyoruz

            # Modeli çalıştırıyoruz (İleri besleme)
            outputs = model(images)

            # Kayıp fonksiyonunu kullanarak loss hesaplıyoruz
            loss = criterion(outputs, labels)
            running_loss += loss.item()  # Toplam kaybı biriktiriyoruz

            # Tahminleri alıyoruz
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()  # Doğru tahminleri sayıyoruz
            total_samples += labels.size(0)  # Toplam örnek sayısını güncelliyoruz

    avg_loss = running_loss / len(test_loader)  # Ortalama kaybı hesaplıyoruz
    accuracy = (correct_predictions / total_samples) * 100  # Doğruluk oranı

    return avg_loss, accuracy
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(img_size=28).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
    ])

train_dataset = datasets.MNIST(root="./data",train=True,download=True,transform=transform)
test_dataset = datasets.MNIST(root="./data",train=False,download=True,transform=transform)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

num_epochs = 100
best_accuracy = 0.00

for epoch in range(num_epochs):
    train_loss,train_accuracy = train(model, train_loader, criterion, optimizer, device)
    test_loss,test_accuracy = test(model, test_loader, criterion, device)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%,'
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(),"mnist_cnn.pth")
        print("kaydedildi")
    
