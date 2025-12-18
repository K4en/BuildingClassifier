import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ArchitectureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # All subfolders = class labels
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.classes_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_path = self.root_dir / cls_name
            for img_file in cls_path.glob("*.jpg"):
                self.samples.append((img_file, self.classes_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label