import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import BuildingClassifier
import os

# 1. Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 3. Dataset and Dataloader
train_dataset = datasets.ImageFolder(root='Data/Training', transform=transform)
val_dataset = datasets.ImageFolder(root='Data/Val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. Model
model = BuildingClassifier().to(device)

# 5. Loss function and optimizer
criterion = nn.BCELoss() # Binary cross entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

def binary_target(y):
    return y.float().unsqueeze(1)

# 6. Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = binary_target(labels).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

# 7. Save model
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/buildingclassifier.pth")
print("Model saved to models/buildingclassifier.pth")