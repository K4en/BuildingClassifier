import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from model import BuildingClassifier
import shutil
from PIL import Image
import numpy as np

from train import train_dataset, val_dataset, criterion

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
REPLAY_DIR = "replay_buffer"

# Transform
transform = transforms.Compose([
    transforms.Resize(128, 128),
    transforms.ToTensor(),
])

# Dataset
train_dataset = datasets.ImageFolder(root="Data/Train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Validation dataset
val_dataset = datasets.ImageFolder(root="Data/Val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# Model setup
model = BuildingClassifier().to(device)
if os.path.exists("models/buildingclassifier.pth"):
    model.load_state_dict(torch.load("models/buildingclassifier.pth"))
model.train()

criterion = nn.BCELoss(reduction="none") # to track individual sample losses
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train with replay logic
def save_replay_image(image_tensor, class_idx, filename):
    os.makedirs(f"{REPLAY_DIR}/{class_idx}", exist_ok=True)
    image = transforms.ToPILImage()(image_tensor.cpu())
    image.save(os.path.join(REPLAY_DIR, str(class_idx), filename))

for epoch in range(5):
    epoc_losses = [] # store (loss, image, label, idx) tuples

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        losses = criterion(outputs, labels) # shape: [batch_size, 1]
        losses.mean().backward()
        optimizer.step()

        for i in range(images.size(0)):
            loss_val = losses[i].item()
            epoc_losses.append((loss_val, images[i], labels[i].item(),
                                f"epoch{epoch}_batch{batch_idx}_batch_idx{batch_idx}_img{i}.jpg"))

    # Save top & bottom samples
    sorted_losses = sorted(epoc_losses, key=lambda x: x[0])
    best_samples = sorted_losses[:10] # lowest loss
    worst_samples = sorted_losses[-10:] # highest loss

    for loss_val, img_tensor, label, fname in best_samples + worst_samples:
        save_replay_image(img_tensor, int(label), fname)

    print(f"Epoch {epoch+1}: Saved replay samples")

# Save model after training
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/buildingclassifier.pth")
print("Training complete, Model saved")

# Inject replay in next training ground
class ReplayDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.samples = []
        self.transform = transform
        for label_dir in os.listdir(root):
            class_path = os.path.join(root, label_dir)
            for img_name in os.listdir(class_path):
                self.samples.append((os.path.join(class_path, img_name), int(label_dir)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(label)

# Combine replay + current data
if os.path.exists(REPLAY_DIR):
    replay_dataset = ReplayDataset(REPLAY_DIR, transform=transform)
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, replay_dataset])
    train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
    print(f"Replay buffer loaded: {len(replay_dataset)} samples")


