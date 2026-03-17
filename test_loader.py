print("Starting test...")

from preprocessing.dataset_loader import HockeyDataset

print("Import successful")

dataset = HockeyDataset(
    r"D:\MAJOR PROJECT\violence_project\data\hockey\train"
)

print("Dataset length:", len(dataset))

frames, label = dataset[0]

print("Frame shape:", frames.shape)
print("Label:", label)
