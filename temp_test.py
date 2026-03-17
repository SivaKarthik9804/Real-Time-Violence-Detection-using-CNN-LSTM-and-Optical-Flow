from preprocessing.dataset_loader import HockeyDataset

train_dataset = HockeyDataset(
    r"D:\MAJOR PROJECT\violence_project\data\hockey\train"
)

print("Total samples:", len(train_dataset))

frames, label = train_dataset[0]

print("Frame shape:", frames.shape)
print("Label:", label)
