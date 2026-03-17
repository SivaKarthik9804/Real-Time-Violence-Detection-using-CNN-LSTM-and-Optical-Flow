import os

base = r"D:\MAJOR PROJECT\violence_project\data\hockey"

print("Train Fight:", len(os.listdir(os.path.join(base, "train", "fight"))))
print("Train NonFight:", len(os.listdir(os.path.join(base, "train", "nonfight"))))
print("Val Fight:", len(os.listdir(os.path.join(base, "val", "fight"))))
print("Val NonFight:", len(os.listdir(os.path.join(base, "val", "nonfight"))))
