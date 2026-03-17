import streamlit as st
import torch
import cv2
import numpy as np
import time
import os
from models.mobilenet_lstm import ViolenceModel

# ---------------- CONFIG ----------------
MODEL_PATH = r"D:\MAJOR PROJECT\violence_project\weights\best_cpu_model.pth"
THRESHOLD = 0.35
NUM_FRAMES = 8

device = torch.device("cpu")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = ViolenceModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

# ---------------- FRAME EXTRACTION ----------------
def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("❌ Error opening video file")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        st.error("❌ Video contains 0 frames")
        return None

    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    if len(frames) == 0:
        st.error("❌ No frames extracted")
        return None

    frames = np.array(frames) / 255.0
    frames = torch.tensor(frames).permute(0, 3, 1, 2).float()

    return frames.unsqueeze(0)

# ---------------- UI ----------------
st.title("🚨 Violence Detection System (CPU Optimized)")
st.write("Upload a video to detect violence.")

uploaded_file = st.file_uploader("Upload a video", type=["avi", "mp4"])

if uploaded_file is not None:

    # Save temp file
    temp_path = "temp_video.avi"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(temp_path)

    if st.button("Analyze Video"):

        model = load_model()

        st.write("Processing...")
        start_time = time.time()

        frames = extract_frames(temp_path, NUM_FRAMES)

        if frames is None:
            st.stop()

        frames = frames.to(device)

        with torch.no_grad():
            output = model(frames)
            probability = output.item()

        end_time = time.time()

        # Show raw probability
        st.write(f"Raw Probability: {probability:.4f}")
        st.write(f"Inference Time: {end_time - start_time:.3f} seconds")

        # Classification
        if probability > THRESHOLD:
            st.error("🚨 VIOLENCE DETECTED")
        else:
            st.success("✅ No Violence Detected")

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
