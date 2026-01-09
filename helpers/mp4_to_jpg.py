import cv2
import os
import sys

video_path = r"C:\Users\andre\Desktop\record_089 - Trim.mp4"
output_dir = r"C:\Users\andre\Desktop\chess_frames_less"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть видео")

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % 10 == 0:
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    frame_idx += 1

cap.release()
print(f"Готово. Сохранено кадров: {frame_idx}")
