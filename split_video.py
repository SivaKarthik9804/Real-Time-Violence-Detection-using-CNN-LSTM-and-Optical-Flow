import cv2
import os

def split_video_into_clips(video_path, output_dir, clip_duration_sec=2, label="violence"):
    
    # Create label folder inside output directory
    label_folder = os.path.join(output_dir, label)
    os.makedirs(label_folder, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_clip = fps * clip_duration_sec

    print(f"FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    print(f"Frames per clip: {frames_per_clip}")

    clip_count = 0
    frame_count = 0

    while True:
        clip_frames = []

        for _ in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break
            clip_frames.append(frame)

        if len(clip_frames) == 0:
            break

        clip_filename = os.path.join(label_folder, f"{label}_clip_{clip_count}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, _ = clip_frames[0].shape
        out = cv2.VideoWriter(clip_filename, fourcc, fps, (width, height))

        for frame in clip_frames:
            out.write(frame)

        out.release()
        print(f"Saved: {clip_filename}")

        clip_count += 1

    cap.release()
    print("Done splitting video.")


if __name__ == "__main__":
    
    video_path = r"D:\MAJOR PROJECT\violence_project\input_video\fight_video.mp4"
    output_dir = "output_clips"
    clip_duration_sec = 2   # change to 3 if needed
    label = "violence"      # change to "non_violence" if needed

    split_video_into_clips(video_path, output_dir, clip_duration_sec, label)
