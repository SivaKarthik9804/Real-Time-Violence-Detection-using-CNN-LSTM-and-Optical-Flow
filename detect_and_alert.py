import torch
import cv2
import numpy as np
import time
from models.mobilenet_lstm import ViolenceModel

# -------- CONFIG --------
VIDEO_PATH = r"D:\MAJOR PROJECT\violence_project\data\hockey\train\fight\fi1_xvid.avi"
MODEL_PATH = r"D:\MAJOR PROJECT\violence_project\weights\best_cpu_model.pth"
THRESHOLD = 0.5
NUM_FRAMES = 8

device = torch.device("cpu")

# -------- FRAME EXTRACTION --------
def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    frames = np.array(frames) / 255.0
    frames = torch.tensor(frames).permute(0, 3, 1, 2).float()

    return frames.unsqueeze(0)

# -------- LOAD MODEL --------
model = ViolenceModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -------- PREDICTION --------
print("Analyzing video...")
start_time = time.time()

frames = extract_frames(VIDEO_PATH, NUM_FRAMES).to(device)

with torch.no_grad():
    output = model(frames)
    probability = output.item()

end_time = time.time()

print(f"Violence Probability: {probability:.4f}")
print(f"Inference Time: {end_time - start_time:.3f} seconds")

if probability > THRESHOLD:
    print("🚨 VIOLENCE DETECTED 🚨")
else:
    print("✅ No Violence Detected")
