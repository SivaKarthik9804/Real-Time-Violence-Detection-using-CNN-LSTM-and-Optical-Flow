import os
import shutil

# ---- PATHS ----
HOCKEY_PATH = r"D:\MAJOR PROJECT\violence_project\data\hockey\train"
NEW_PATH = r"D:\MAJOR PROJECT\violence_project\data\new_dataset"
COMBINED_PATH = r"D:\MAJOR PROJECT\violence_project\data\combined"

# Create folders
os.makedirs(COMBINED_PATH + "\\fight", exist_ok=True)
os.makedirs(COMBINED_PATH + "\\nonfight", exist_ok=True)

def copy_videos(source_path, class_name):
    files = os.listdir(source_path)
    for file in files:
        shutil.copy(
            os.path.join(source_path, file),
            os.path.join(COMBINED_PATH, class_name, file)
        )

# Copy hockey
copy_videos(HOCKEY_PATH + "\\fight", "fight")
copy_videos(HOCKEY_PATH + "\\nonfight", "nonfight")

# Copy new dataset
copy_videos(NEW_PATH + "\\fight", "fight")
copy_videos(NEW_PATH + "\\nonfight", "nonfight")

print("Datasets merged successfully.")
