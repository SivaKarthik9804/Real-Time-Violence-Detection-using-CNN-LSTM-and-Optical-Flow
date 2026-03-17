import cv2
import torch
import numpy as np
import requests
import time
from collections import deque
from models.mobilenet_lstm import ViolenceModel

# ---------------- CONFIG ----------------
MODEL_PATH = r"D:\MAJOR PROJECT\violence_project\weights\best_cpu_model.pth"

THRESHOLD = 0.6
MOTION_THRESHOLD = 2.0
NUM_FRAMES = 16
SMOOTHING_WINDOW = 3
ALERT_COOLDOWN = 10  # seconds

# ---- TELEGRAM CONFIG ----
TELEGRAM_BOT_TOKEN = "8526360233:AAFXnOiCjCIeioJrsLpPTjApfju6Kzu5np0"
TELEGRAM_CHAT_ID = "1359791476"

device = torch.device("cpu")

# ---------------- LOAD MODEL ----------------
model = ViolenceModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------------- BUFFERS ----------------
frame_buffer = deque(maxlen=NUM_FRAMES)
prediction_history = deque(maxlen=SMOOTHING_WINDOW)

last_alert_time = 0

# ---------------- PREPROCESS ----------------
def preprocess_frames(frames):
    frames = np.array(frames) / 255.0
    frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
    return frames.unsqueeze(0)

# ---------------- OPTICAL FLOW ----------------
def compute_optical_flow_magnitude(frames):
    total_magnitude = 0

    for i in range(len(frames) - 1):
        prev = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        nxt = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev, nxt,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        total_magnitude += np.mean(magnitude)

    return total_magnitude / (len(frames) - 1)

# ---------------- TELEGRAM ALERT ----------------
def send_telegram_alert(image_path, probability, motion):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

    caption = (
        f"🚨 VIOLENCE DETECTED\n"
        f"Probability: {probability:.2f}\n"
        f"Motion Score: {motion:.2f}\n"
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    with open(image_path, "rb") as photo:
        requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
            files={"photo": photo}
        )

# ---------------- START CAMERA ----------------
cap = cv2.VideoCapture(0)
print("System started. Press Q to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    frame_buffer.append(rgb)

    if len(frame_buffer) == NUM_FRAMES:

        # ---- Deep Model Prediction ----
        input_tensor = preprocess_frames(frame_buffer).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()


        prediction_history.append(prob)
        avg_prob = sum(prediction_history) / len(prediction_history)

        # ---- Optical Flow Motion Score ----
        motion_score = compute_optical_flow_magnitude(list(frame_buffer))

        # ---- Dual Verification Logic ----
        if avg_prob > THRESHOLD and motion_score > MOTION_THRESHOLD:
            label = "VIOLENCE"
            color = (0, 0, 255)
        else:
            label = "SAFE"
            color = (0, 255, 0)

        # ---- Display Info ----
        cv2.putText(frame, f"{label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 3)

        cv2.putText(frame, f"Prob: {avg_prob:.2f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255,255,0), 2)

        cv2.putText(frame, f"Motion: {motion_score:.2f}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255,255,0), 2)

        # ---- Alert Trigger ----
        current_time = time.time()

        if (avg_prob > THRESHOLD and
            motion_score > MOTION_THRESHOLD and
            current_time - last_alert_time > ALERT_COOLDOWN):

            screenshot_path = "alert.jpg"
            cv2.imwrite(screenshot_path, frame)

            send_telegram_alert(screenshot_path, avg_prob, motion_score)
            print("🚨 Telegram Alert Sent")

            last_alert_time = current_time

    cv2.imshow("Violence Detection + Optical Flow", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
