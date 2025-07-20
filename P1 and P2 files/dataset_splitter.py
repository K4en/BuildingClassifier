import os, shutil, random

def split_dataset(source_root, train_root, val_root, split_ratio=0.8):
    for class_name in os.listdir(source_root):
        class_path = os.path.join(source_root, class_name)
        if not os.path.isdir(class_path):
            continue

        all_images = os.listdir(class_path)
        random.shuffle(all_images)

        split = int(len(all_images) * split_ratio)
        train_imgs = all_images[:split]
        val_imgs = all_images[split:]

        train_class_path = os.path.join(train_root, class_name)
        val_class_path = os.path.join(val_root, class_name)

        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(val_class_path, exist_ok=True)

        for img in train_imgs:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(train_class_path, img)
            )

        for img in val_imgs:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(val_class_path, img)
            )

split_dataset("Data/Raw", "Data/Training", "Data/Val")
split_dataset("Data/Raw", "Data/Training", "Data/Val")