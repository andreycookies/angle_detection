import random

import cv2
import numpy as np
import json
import glob
import os

# =========================
# НАСТРОЙКИ
# =========================
FRAMES_DIR = r"C:\Users\andre\Desktop\chess_frames_less"
OUTPUT_JSON = "video_homo.json"

# 8x10 квадратов → 7x9 внутренних углов
PATTERN_SIZE = (6, 9)
SQUARE_SIZE = 50  # размер квадрата в пикселях
RANSAC_THRESH = 0.8
CROP_BORDER = 0  # 0 = не убирать, 1 = убрать 1 ряд по краям

cols, rows = PATTERN_SIZE  # cols=9, rows=6

# =========================
# ПОДГОТОВКА МИРОВЫХ ТОЧЕК
# =========================
objp = np.zeros((cols * rows, 3), np.float32)  # 3D точки (Z=0)
xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
objp[:, 0] = xx.ravel() * SQUARE_SIZE
objp[:, 1] = yy.ravel() * SQUARE_SIZE

if CROP_BORDER > 0:
    mask = (
        (objp[:, 0] >= CROP_BORDER) &
        (objp[:, 0] < (cols - CROP_BORDER) * SQUARE_SIZE) &
        (objp[:, 1] >= CROP_BORDER) &
        (objp[:, 1] < (rows - CROP_BORDER) * SQUARE_SIZE)
    )
    objp_cropped = objp[mask]
else:
    objp_cropped = objp

# =========================
# СБОР ТОЧЕК
# =========================
all_obj_pts = []  # 3D точки
all_img_pts = []  # 2D точки на изображении

images = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.*")))
print(f"Найдено кадров: {len(images)}")

used_frames = 0
img_shape = None

random.shuffle(images)

for path in images[0:40]:
    img = cv2.imread(path)
    if img is None:
        continue
    if img_shape is None:
        img_shape = img.shape[:2][::-1]  # (width, height)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        PATTERN_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not found:
        continue

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ).reshape(-1, 2)

    if CROP_BORDER > 0:
        corners = corners[mask]

    all_obj_pts.append(objp_cropped)
    all_img_pts.append(corners)
    used_frames += 1

print(f"Использовано кадров: {used_frames}")
if used_frames < 5:
    raise RuntimeError("Недостаточно кадров с найденной шахматкой")

# =========================
# КАЛИБРОВКА КАМЕРЫ
# =========================
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    all_obj_pts, all_img_pts, img_shape, None, None
)

print("Матрица камеры K:\n", K)
print("Коэффициенты дисторсии dist:\n", dist.ravel())

# =========================
# ОДНА СРЕДНЯЯ ГОМOГРАФИЯ (через все точки)
# =========================
obj_pts_2d = np.vstack([pts[:, :2] for pts in all_obj_pts])  # берем X,Y
img_pts_2d = np.vstack(all_img_pts)

H, inliers = cv2.findHomography(obj_pts_2d, img_pts_2d, cv2.RANSAC, RANSAC_THRESH)
if H is None:
    raise RuntimeError("Гомография не найдена")

H = H / H[2, 2]  # нормализация

# =========================
# РЕПОЗИЦИЯ И ОШИБКА
# =========================
proj = cv2.perspectiveTransform(obj_pts_2d.reshape(-1,1,2), H).reshape(-1,2)
errors = np.linalg.norm(proj - img_pts_2d, axis=1)
stats = {
    "mean_px": float(np.mean(errors)),
    "median_px": float(np.median(errors)),
    "p95_px": float(np.percentile(errors, 95)),
    "used_frames": used_frames,
    "total_points": int(len(errors))
}
print("Ошибка репроекции (px):", stats)

# =========================
# СОХРАНЕНИЕ В JSON
# =========================
H_inv = np.linalg.inv(H)
H_inv = H_inv / H_inv[2,2]

output = {
    "homography_matrix": H_inv.tolist(),
    "camera_matrix": K.tolist(),
    "distortion_coefficients": dist.ravel().tolist(),
    "reprojection_error_px": stats,
    "pattern_size": {
        "inner_corners": [cols, rows],
        "squares": [cols + 1, rows + 1]
    }
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)

print(f"Гомография и калибровка сохранены в {OUTPUT_JSON}")
