import os

base = r"D:\MAJOR PROJECT\violence_project\data\combined_split"

print("Train Fight:", len(os.listdir(base + "\\train\\fight")))
print("Train NonFight:", len(os.listdir(base + "\\train\\nonfight")))
print("Val Fight:", len(os.listdir(base + "\\val\\fight")))
print("Val NonFight:", len(os.listdir(base + "\\val\\nonfight")))
