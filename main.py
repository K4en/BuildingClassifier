from model import ArchitectureClassifier
from dataset import ArchitectureDataset
from train import train_model
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from replay_dataset import ReplayDataset, get_focus_loader

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = ArchitectureDataset("Data/Phase 3/Training", transform=transform)
    val_dataset = ArchitectureDataset("Data/Phase 3/Val", transform=transform)
    replay_focus = ReplayDataset("replay/focus.json", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    replay_loader = DataLoader(replay_focus, batch_size=4, shuffle=True)

    model = ArchitectureClassifier(num_classes=5)

    print(" Starting training loop...")
    train_model(model, train_loader, val_loader, num_epochs=10, val_dataset=val_dataset, replay_loader=replay_loader)

    focus_loader = get_focus_loader(
        dataset_dir="Data/Phase 3/Val",
        json_path="replay/focus.json",
        transform=transform,
        batch_size=16,
    )

    train_model(model, focus_loader, val_loader=None, num_epochs=3, lr=1e-4)

if __name__ == "__main__":
    main()