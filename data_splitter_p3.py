import os
import random
import shutil
from pathlib import Path

RAW_DIR = Path("Data/Phase 3/Raw")
TRAIN_DIR = Path("Data/Phase 3/Training")
VAL_DIR = Path("Data/Phase 3/Val")
SPLIT_RATIO = 0.8

def split_dataset():
    for culture_folder in RAW_DIR.iterdir():
        if not culture_folder.is_dir():
            continue

        images = list(culture_folder.glob("*.jpg"))
        random.shuffle(images)

        split_index = int(len(images) * SPLIT_RATIO)
        train_images = images[:split_index]
        val_images = images[split_index:]

        train_target = TRAIN_DIR / culture_folder.name
        val_target = VAL_DIR / culture_folder.name
        train_target.mkdir(parents=True, exist_ok=True)
        val_target.mkdir(parents=True, exist_ok=True)

        for img_path in train_images:
            shutil.copy(img_path, train_target / img_path.name)

        for img_path in val_images:
            shutil.copy(img_path, val_target / img_path.name)

        print(f"{culture_folder.name}: {len(train_images)} train / {len(val_images)} val")

if __name__ == "__main__":
    split_dataset()