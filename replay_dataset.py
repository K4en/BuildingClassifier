from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
from PIL import Image
import json
import os

def get_focus_loader(dataset_dir, json_path, transform, batch_size=16):
    # Load full dataset
    full_dataset = ImageFolder(dataset_dir, transform=transform)

    # Normalize paths from json
    with open(json_path, 'r') as f:
        data = json.load(f)

    focus_paths = set([
        os.path.normpath(os.path.relpath(item["image_path"], start=dataset_dir))
        for item in data
    ])

    # Normalize paths in dataset too
    indices = [i for i, (path, _) in enumerate(full_dataset.samples)
               if os.path.normpath(os.path.relpath(path, start=dataset_dir))
               in focus_paths]

    print(f"✅ Focus loader will use {len(indices)} samples from {json_path}")
    if len(indices) == 0:
        raise ValueError("No matching samples found for focus loader!")

    subset = Subset(full_dataset, indices)

    return DataLoader(subset, batch_size=batch_size, shuffle=True)

class ReplayDataset(Dataset):
    def __init__(self, path, transform=None):
        self.samples = []
        self.transform = transform

        with open(path, 'r') as f:
            data = json.load(f)
            for entry in data:
                img_path = entry.get('image_path')
                label = entry.get('true_label')
                if img_path is not None:
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label