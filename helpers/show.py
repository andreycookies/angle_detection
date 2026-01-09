import cv2
import numpy as np
import json

# Пути
video_path = r"C:\Users\Администратор\Desktop\завод\record_090 - Trimss.mp4"
homography_path = r"C:\Users\Администратор\Desktop\завод\shittens\video_homo.json"

WINDOW_NAME = "Homography"

# ====== ЗАГРУЗКА ГОМOГРАФИИ ======
with open(homography_path, "r", encoding="utf-8") as f:
    data = json.load(f)

H = np.array(data["homography_matrix"], dtype=np.float32)

# ====== ОТКРЫТИЕ ВИДЕО ======
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть видео")

# ====== РАЗМЕР ЭКРАНА ======
screen_w = 1920
screen_h = 1080

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ====== ПОЛУЧЕНИЕ ГРАНИЦ КАДРА ПОСЛЕ ГОМOГРАФИИ ======
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32).reshape(-1, 1, 2)  # shape (4,1,2)

    warped_corners = cv2.perspectiveTransform(corners, H)
    x_coords = warped_corners[:, 0, 0]
    y_coords = warped_corners[:, 0, 1]

    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()

    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))

    # ====== СДВИГ, ЧТОБЫ ВСЁ ПОПАЛО В КАДР ======
    translation = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ], dtype=np.float32)

    H_translate = translation @ H

    warped = cv2.warpPerspective(frame, H_translate, (new_w, new_h))

    # ====== МАСШТАБИРОВАНИЕ ПОД ЭКРАН ======
    scale = min(screen_w / new_w, screen_h / new_h, 1.0)
    resized_w = int(new_w * scale)
    resized_h = int(new_h * scale)

    warped_resized = cv2.resize(warped, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    cv2.imshow(WINDOW_NAME, warped_resized)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()