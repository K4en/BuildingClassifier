import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import BuildingClassifier
import os

# Config #
MODEL_PATH = "models/buildingclassifier.pth"
DATA_DIR = "Data/Val"
BATCH_SIZE = 32

# Transformations #
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load validation dataset #
val_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model #
model = BuildingClassifier()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Eval loop #
total = 0
correct = 0

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        predicted = torch.round(outputs).squeeze()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")