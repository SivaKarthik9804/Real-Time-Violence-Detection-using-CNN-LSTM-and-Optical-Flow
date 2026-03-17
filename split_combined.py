import os
import random
import shutil

SOURCE_PATH = r"D:\MAJOR PROJECT\violence_project\data\combined"
TARGET_PATH = r"D:\MAJOR PROJECT\violence_project\data\combined_split"

TRAIN_RATIO = 0.8

# Create target folders
for split in ["train", "val"]:
    for cls in ["fight", "nonfight"]:
        os.makedirs(os.path.join(TARGET_PATH, split, cls), exist_ok=True)

def split_class(class_name):
    files = os.listdir(os.path.join(SOURCE_PATH, class_name))
    random.shuffle(files)

    split_index = int(len(files) * TRAIN_RATIO)
    train_files = files[:split_index]
    val_files = files[split_index:]

    for file in train_files:
        shutil.copy(
            os.path.join(SOURCE_PATH, class_name, file),
            os.path.join(TARGET_PATH, "train", class_name, file)
        )

    for file in val_files:
        shutil.copy(
            os.path.join(SOURCE_PATH, class_name, file),
            os.path.join(TARGET_PATH, "val", class_name, file)
        )

split_class("fight")
split_class("nonfight")

print("Combined dataset split successfully.")
