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
    

def create_rect_from_mask_perimeter(mask: np.ndarray, target_ratio = 1):
    mask2 = cv2.imread(r'template_masks\mask_w585_h585.png', cv2.IMREAD_GRAYSCALE)
    return  (mask2[:,:,0] > 0).astype(np.uint8)


def get_max_radius(mask, cx, cy):
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        return 0
    dists = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    return int(np.max(dists))



def normalize_masks(mask1, mask2):
    
    
    # Находим центры масс и площади
     
    cv2.imwrite("img/mask01.png", mask1*255)
    
    cv2.imwrite("img/mask02.png", mask2*255)
    m1 = cv2.moments(mask1)
    m2 = cv2.moments(mask2)
    
    c1 = (int(m1['m10'] / m1['m00']), int(m1['m01'] / m1['m00']))
    c2 = (int(m2['m10'] / m2['m00']), int(m2['m01'] / m2['m00']))
    
    area1, area2 = m1['m00'], m2['m00']
    
    # Масштабируем меньшую маску
    scale = np.sqrt(max(area1, area2) / min(area1, area2))
    if area1 < area2:
        mask1 = cv2.resize(mask1, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        m1 = cv2.moments(mask1)
        c1 = (int(m1['m10'] / m1['m00']), int(m1['m01'] / m1['m00']))
    else:
        mask2 = cv2.resize(mask2, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        m2 = cv2.moments(mask2)
        c2 = (int(m2['m10'] / m2['m00']), int(m2['m01'] / m2['m00']))


    
    cv2.imwrite("img/mask1.png", mask1*255)
    
    cv2.imwrite("img/mask2.png", mask2*255)
    # Определяем размер выходного изображения
    h = max(mask1.shape[0], mask2.shape[0]) + max(c1[1], c2[1])
    w = max(mask1.shape[1], mask2.shape[1]) + max(c1[0], c2[0])

    target_size = max(w, h)
    if target_size % 2 != 0: target_size += 1
    
    # Центрируем обе маски
    canvas1 = np.zeros((target_size, target_size), dtype=np.uint8)
    canvas2 = np.zeros((target_size, target_size), dtype=np.uint8)
    
    y1, x1 = target_size // 2 - c1[1], target_size // 2 - c1[0]
    y2, x2 = target_size // 2 - c2[1], target_size // 2 - c2[0]
    
    canvas1[y1:y1+mask1.shape[0], x1:x1+mask1.shape[1]] = mask1
    canvas2[y2:y2+mask2.shape[0], x2:x2+mask2.shape[1]] = mask2
    
    return canvas1, canvas2



def calculate_angle(warped_mask: np.ndarray, template_mask: np.ndarray) -> float:

    # 3. Создаем новые src и dst, помещая кропы в центр черного квадрата
    src = np.float32(warped_mask)
    cv2.imwrite("img/src.png", src*255)

    dst = np.float32(template_mask)
    cv2.imwrite("img/dst.png", dst*255)
    # --- КОНЕЦ ВСТАВКИ ---

    # 2. Находим центр и переводим в полярные координаты (Log-Polar)
    # Это превращает поворот в сдвиг по оси Y
    M_src = cv2.moments(src)
    cx_src = M_src['m10'] / (M_src['m00'] + 1e-5)
    cy_src = M_src['m01'] / (M_src['m00'] + 1e-5)
    
    M_dst = cv2.moments(dst)
    cx_dst = M_dst['m10'] / (M_dst['m00'] + 1e-5)
    cy_dst = M_dst['m01'] / (M_dst['m00'] + 1e-5)

    # 4. Linear Polar (для чистого вращения)
    # max_radius берем чуть меньше половины, чтобы не цеплять углы
    radius = max(get_max_radius(src, cx_src, cy_src), 
             get_max_radius(dst, cx_dst, cy_dst))


    # Флаг WARP_FILL_OUTLIERS заполняет "пустоты" нулями
    src_polar = cv2.linearPolar(np.float32(src), (cx_src, cy_src), radius, cv2.WARP_FILL_OUTLIERS)
    dst_polar = cv2.linearPolar(np.float32(dst), (cx_dst, cy_dst), radius, cv2.WARP_FILL_OUTLIERS)

    # 3. Фазовая корреляция находит сдвиг
    hann = cv2.createHanningWindow(src_polar.shape[:2][::-1], cv2.CV_32F)
    shift, _ = cv2.phaseCorrelate(src_polar, dst_polar, window=hann)

    # 6. Расчет угла
    dy = shift[1]
    angle = -(dy * 360.0) / src_polar.shape[0] # shape[0] это высота (h)
    angle = min(np.abs(90-np.abs(angle)), np.abs(angle))

    return float(angle)


def detect_sheet_angle_no_homography(warped_mask: np.ndarray) -> float:



    warped_mask = mask_preprocessing(warped_mask)

    template_mask = create_rect_from_mask_perimeter(warped_mask)

    x, y, w_c, h_c = cv2.boundingRect(cv2.findNonZero(warped_mask))
    warped_mask_cropped = warped_mask[y:y+h_c, x:x+w_c]

    x2, y2, w_c2, h_c2 = cv2.boundingRect(cv2.findNonZero(template_mask))
    template_mask_cropped = template_mask[y2:y2+h_c2, x2:x2+w_c2]

    norm_warped_mask, norm_template_mask = normalize_masks(warped_mask_cropped, template_mask_cropped)

    return calculate_angle(norm_warped_mask, norm_template_mask)
