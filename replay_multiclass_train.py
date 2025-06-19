import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
from model import BuildingClassifier
from PIL import Image
from utils import plot_training_loss

# Config #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
REPLAY_DIR = "replay_buffer"

# Transform #
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Datasets #
train_dataset = datasets.ImageFolder(root="Data/Training", transform=transform)
val_dataset = datasets.ImageFolder(root="Data/Val", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# Model setup #
num_classes = len(train_dataset.classes)
model = BuildingClassifier(num_classes=num_classes).to(device)
if os.path.exists("models/buildingmulticlassifier.pth"):
    model.load_state_dict(torch.load("models/buildingmulticlassifier.pth"))
model.train()
criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Replay Saving #
def save_replay_image(image_tensor, class_idx, filename):
    os.makedirs(f"{REPLAY_DIR}/{class_idx}", exist_ok=True)
    image = transforms.ToPILImage()(image_tensor.cpu())
    image.save(os.path.join(REPLAY_DIR, str(class_idx), filename))

loss_values = []

for epoch in range(5):
    epoc_losses = []
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        losses = criterion(outputs, labels)
        losses.mean().backward()
        optimizer.step()

        for i in range(images.size(0)):
            epoc_losses.append((losses[i].item(), images[i], labels[i].item(),
                               f"epoch{epoch}_batch{batch_idx}_img{i}.jpg"))

    sorted_losses = sorted(epoc_losses, key=lambda x: x[0])
    best_samples = sorted_losses[:10]
    worst_samples = sorted_losses[-10:]

    for loss_val, img_tensor, label, fname in best_samples + worst_samples:
        save_replay_image(img_tensor, int(label), fname)

    avg_loss = sum(x[0] for x in epoc_losses) / len(epoc_losses)
    loss_values.append(avg_loss)
    print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}. Saved replay samples")

# Save model #
os.makedirs("replay_buffer", exist_ok=True)
torch.save(model.state_dict(), "models/buildingmulticlassifier.pth")
print("Training complete, Model save! Banzai it worked!")

# Replay loader
class ReplayDataset(Dataset):
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
        return self.transform(image), label

# Merge replay if exists
if os.path.exists(REPLAY_DIR):
    replay_dataset = ReplayDataset(REPLAY_DIR, transform=transform)
    combined_dataset = ConcatDataset([train_dataset, replay_dataset])
    train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
    print(f"Replay buffer loaded: {len(replay_dataset)} samples")

plot_training_loss(loss_values, save_path="plots/multi_training_loss.png")