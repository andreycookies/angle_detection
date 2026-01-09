import cv2
import numpy as np


def mask_preprocessing(mask: np.ndarray):
    m = mask.copy()
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

        # Морфологические операции — меньше итераций, адаптивно
        k = max(3, int(round(min(m.shape[:2]) / 100)))
        kernel = np.ones((k, k), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Вместо размытия — контурное сглаживание
        m = cv2.dilate(m, kernel, iterations=1)
        m = cv2.erode(m, kernel, iterations=1)
    return m


def order_pts(pts):
    """Упорядочивает точки квадрата в формат tl tr br bl"""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_4_points(mask: np.ndarray):
    """Апроксимирует контур к четырехугольнику, возвращает координаты точек"""
    m = mask.copy()
    cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if not cnts:
        return float("nan")
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:  # слишком маленькая область
        return float("nan")

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.001 * peri, True)
    approx4 = order_pts(approx.reshape(len(approx), 2)).reshape(4, 1, 2)
    return approx4, peri


def has_no_ones(array):
    """Возвращает True, если в массиве все биты - нулевые"""
    # Если это OpenCV Mat, преобразуем в ndarray для единообразия
    if hasattr(array, 'dtype'):  # Работает и для Mat и для ndarray
        return not np.any(array == 1)
    else:
        # Для других типов сначала преобразуем в ndarray
        arr = np.array(array)
        return not np.any(arr == 1)


def detect_sheet_angle_no_homography(warped_mask: np.ndarray) -> float:
    warped_mask = mask_preprocessing(warped_mask)

    approx4_out, _ = find_4_points(warped_mask)

    data = approx4_out.reshape(-1, 2).astype(np.float64)

    x_vec = (data[0, 0] + data[3, 0]) / 2 - (data[1, 0] + data[2, 0]) / 2
    y_vec = (data[0, 1] + data[3, 1]) / 2 - (data[1, 1] + data[2, 1]) / 2
    v = [x_vec, y_vec]
    angle_rad = np.arctan2(v[1], v[0])
    angle_deg = np.degrees(angle_rad)
    # угол отклонения от вертикали: вертикаль = +90° (atan2(1,0)=90°)
    deviation = angle_deg - 90.0
    # нормализация в [-90,90]
    deviation = ((deviation + 180) % 180) - 90

    return deviation
