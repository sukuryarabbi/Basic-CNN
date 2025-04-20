import torch
import cv2
import numpy as np
from torchvision import transforms
from model import CNN  


img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = 255 - img


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img_tensor = transform(img).unsqueeze(0)    # (1, 1, 28, 28) şeklinde batch haline getir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(img_size=28).to(device)

model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

with torch.no_grad():
    img_tensor = img_tensor.to(device)
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    print(f"Tahmin edilen rakam: {predicted.item()}")
