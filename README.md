# Real-Time-Violence-Detection-using-CNN-LSTM-and-Optical-Flow
AI-based violence detection system using CNN-LSTM and Optical Flow to capture spatio-temporal and motion patterns in video streams for real-time surveillance.
📌 Overview

This project presents a real-time violence detection system designed for intelligent surveillance using deep learning and edge computing.

The system integrates:

CNN for spatial feature extraction

LSTM for temporal sequence modeling

Optical Flow for motion analysis

Additionally, it is deployed in a drone-based surveillance setup using a Raspberry Pi, enabling wide-area monitoring and real-time incident detection.

🚀 Key Highlights

Real-time violence detection from live video streams

CNN-LSTM architecture for spatio-temporal learning

Optical Flow for motion pattern recognition

Edge deployment on Raspberry Pi (CPU-based inference)

Drone-based mobile surveillance system

Automatic evidence capture + live streaming

🧠 System Architecture
🔹 Processing Pipeline

Video Input (Drone Camera / CCTV)

Frame Extraction & Preprocessing

Optical Flow Computation (Motion Features)

CNN Feature Extraction

LSTM Temporal Modeling

Classification (Violence / Non-Violence)

Alert + Evidence Storage

📊 Results
✅ Performance Metrics

Accuracy: 88%

Precision: 0.82

Recall (Violence): 0.96

F1-Score: 0.88

ROC-AUC: 0.958

🔍 Confusion Matrix
	Predicted Non-Violence	Predicted Violence
Actual Non-Violence	80 (TN)	21 (FP)
Actual Violence	4 (FN)	96 (TP)
📈 Model Comparison
Model	Accuracy	Precision	Recall	F1 Score	ROC-AUC
CNN	83%	0.78	0.88	0.83	0.93
CNN + LSTM + Optical Flow	88%	0.82	0.96	0.88	0.958

👉 Optical Flow + LSTM significantly improves temporal understanding and motion sensitivity, reducing missed detections.

📂 Dataset Used

RLVS (Real-Life Violence Situations) – real-world outdoor scenes (primary dataset)

Hockey Fight Dataset – limited relevance

RWF-2000 – mixed realism

🛠 Tech Stack

Python

TensorFlow / Keras

OpenCV (Optical Flow)

NumPy

Raspberry Pi OS

⚙️ Hardware Setup

Quadcopter Drone

Raspberry Pi

Raspberry Pi Camera Module

Onboard Battery

Ground Monitoring Station

🎯 Applications

Smart campus surveillance

Crime detection systems

Public safety monitoring

Anti-ragging systems

Security patrol automation

⚠️ Limitations

False positives in crowded scenes

Performance depends on dataset quality

Limited FPS on CPU-based edge devices

Lighting and occlusion challenges

🔮 Future Improvements

Deploy on Jetson Nano / Edge TPU for better performance

Fully autonomous drone navigation

Real-time alert system (SMS / App integration)

Multi-camera fusion system

Transformer-based video models (next-gen upgrade)

▶️ How to Run
git clone https://github.com/your-username/violence-detection.git
cd violence-detection
pip install -r requirements.txt
python main.py
📌 Key Insight (Don’t Ignore This)

Most projects stop at “model training.”
This one goes further:

✔ Motion modeling (Optical Flow)
✔ Temporal learning (LSTM)
✔ Edge deployment (Raspberry Pi)
✔ Real-world system (Drone integration)

That combination is what makes it actually valuable.
