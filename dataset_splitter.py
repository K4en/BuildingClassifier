import os, shutil, random

def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.8):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in os.listdir(source_dir):
        all_images = os.listdir(os.path.join(source_dir, class_name))
        random.shuffle(all_images)

        split = int(len(all_images) * split_ratio)
        train_imgs = all_images[:split]
        val_imgs = all_images[split:]

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        for img in train_imgs:
            shutil.copy(
                os.path.join(source_dir, class_name, img),
                os.path.join(train_dir, class_name, img)
            )

split_dataset("Data/Raw/Residential", "Data/Train/Residential", "Data/Val/Residential")
split_dataset("Data/Raw/Industrial", "Data/Train/Industrial", "Data/Val/Industrial")