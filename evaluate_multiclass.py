import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import BuildingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os

# Config #
VAL_DIR = "Data/Val"
MODEL_PATH = "models/buildingmulticlassifier.pth"
BATCH_SIZE = 32
IMG_SIZE = 128

# Device #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transform #
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Dataset #
dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model Load #
model = BuildingClassifier(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Evaluation #
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Metrics #
acc = np.mean(np.array(y_true) == np.array(y_pred)) * 100
print(f"Accuracy: {acc:.2f}%")

print("\n Classification Report")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix #
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/confusion_matrix.png")
plt.close()