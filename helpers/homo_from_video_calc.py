import cv2
import numpy as np
import json

# =========================
# НАСТРОЙКИ
# =========================

VIDEO_PATH = r"C:\Users\Администратор\Desktop\завод\output_first_90s.mp4"
OUTPUT_JSON = r"C:\Users\Администратор\Desktop\завод\shittens\video_homo.json"

# ВНУТРЕННИЕ углы (cols, rows)
PATTERN_SIZE = (6, 9)

SQUARE_SIZE = 50      # масштаб произвольный, можно оставить
RANSAC_THRESH = 0.8
CROP_BORDER = 0       # 0 = не убирать

MAX_FRAMES = 300      # сколько кадров взять максимум
FRAME_STEP = 50      # брать каждый N-й кадр

# =========================
# ПОДГОТОВКА ТОЧЕК ПЛОСКОСТИ
# =========================

cols, rows = PATTERN_SIZE

objp = np.zeros((cols * rows, 3), np.float32)
xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
objp[:, 0] = xx.ravel() * SQUARE_SIZE
objp[:, 1] = yy.ravel() * SQUARE_SIZE

if CROP_BORDER > 0:
    mask = (
        (xx.ravel() >= CROP_BORDER) &
        (xx.ravel() < cols - CROP_BORDER) &
        (yy.ravel() >= CROP_BORDER) &
        (yy.ravel() < rows - CROP_BORDER)
    )
    objp_cropped = objp[mask]
else:
    objp_cropped = objp
    mask = None

# =========================
# ЧТЕНИЕ ВИДЕО И СБОР ТОЧЕК
# =========================

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть видео")

all_obj_pts = []
all_img_pts = []

frame_idx = 0
used_frames = 0
img_shape = None

print("Чтение видео...")

while used_frames < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % FRAME_STEP != 0:
        continue

    if img_shape is None:
        img_shape = frame.shape[:2][::-1]  # (w, h)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        PATTERN_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not found:
        continue  # предполагается, что редко, но пусть будет

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

cap.release()

print(f"Использовано кадров: {used_frames}")

if used_frames < 5:
    raise RuntimeError("Недостаточно кадров с шахматкой")

# =========================
# КАЛИБРОВКА КАМЕРЫ
# =========================

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    all_obj_pts,
    all_img_pts,
    img_shape,
    None,
    None
)

print("Матрица камеры K:\n", K)
print("Дисторсия:\n", dist.ravel())

# =========================
# СРЕДНЯЯ ГОМOГРАФИЯ
# =========================

obj_pts_2d = np.vstack([pts[:, :2] for pts in all_obj_pts])
img_pts_2d = np.vstack(all_img_pts)

H, inliers = cv2.findHomography(
    obj_pts_2d,
    img_pts_2d,
    cv2.RANSAC,
    RANSAC_THRESH
)

if H is None:
    raise RuntimeError("Гомография не найдена")

H /= H[2, 2]

# =========================
# ОШИБКА РЕПРОЕКЦИИ
# =========================

proj = cv2.perspectiveTransform(
    obj_pts_2d.reshape(-1, 1, 2),
    H
).reshape(-1, 2)

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
# ОБРАТНАЯ ГОМOГРАФИЯ + JSON
# =========================

H_inv = np.linalg.inv(H)
H_inv /= H_inv[2, 2]

output = {
    "homography_matrix": H_inv.tolist(),
    "camera_matrix": K.tolist(),
    "distortion_coefficients": dist.ravel().tolist(),
    "reprojection_error_px": stats,
    "pattern_size": {
        "inner_corners": [cols, rows]
    }
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)

print(f"Результаты сохранены в {OUTPUT_JSON}")
