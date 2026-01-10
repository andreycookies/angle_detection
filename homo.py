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

    # 2. Находим контур и его периметр
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return
    
    # Берем самый большой контур
    cnt = max(contours, key=cv2.contourArea)
    
    # Считаем периметр (True = замкнутый)
    perimeter = cv2.arcLength(cnt, True)
    
    print(f"Периметр маски: {perimeter:.2f} пикселей")
    
    # 3. Вычисляем размеры нового прямоугольника
    # P = 2 * (w + h)
    # w = ratio * h
    # P = 2 * (ratio*h + h) = 2h * (ratio + 1)
    # h = P / (2 * (ratio + 1))
    
    new_h = perimeter / (2 * (target_ratio + 1))
    new_w = new_h * target_ratio
    
    print(f"Новые размеры: {new_w:.2f} x {new_h:.2f} (Ratio: {target_ratio})")
    
    # 4. Создаем изображение и рисуем прямоугольник по центру
    # Размер холста делаем чуть больше фигуры
    canvas_w = int(new_w*1.5)
    canvas_h = int(new_h*1.5)
    result = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    # Координаты центра
    cx, cy = canvas_w // 2, canvas_h // 2
    
    # Углы прямоугольника (int округление)
    w_int, h_int = int(new_w), int(new_h)
    x1 = cx - w_int // 2
    y1 = cy - h_int // 2
    x2 = x1 + w_int
    y2 = y1 + h_int
    
    # Рисуем (залитый белый)
    return cv2.rectangle(result, (x1, y1), (x2, y2), 255, -1)


def detect_sheet_angle_no_homography(warped_mask: np.ndarray) -> float:
    warped_mask = mask_preprocessing(warped_mask)

    template_mask = create_rect_from_mask_perimeter(warped_mask)

    x, y, w_c, h_c = cv2.boundingRect(cv2.findNonZero(warped_mask))
    warped_mask_cropped = warped_mask[y:y+h_c, x:x+w_c]

    # --- ВСТАВКИ ---
    x2, y2, w_c2, h_c2 = cv2.boundingRect(cv2.findNonZero(template_mask))
    template_mask_cropped = template_mask[y2:y2+h_c2, x2:x2+w_c2]
    
    # 2. Определяем размер общего "холста" (Canvas)
    # Берем максимум по всем измерениям, чтобы оба куска влезли целиком
    # Можно взять фиксированный размер, например 512x512, если объекты не гигантские
    target_size = max(h_c, w_c, h_c2, w_c2) 
    # Лучше сделать размер четным для FFT
    if target_size % 2 != 0: target_size += 1

    def pad_to_size(img, size):
        h, w = img.shape[:2]
        # Вычисляем отступы для центрирования
        top = (size - h) // 2
        bottom = size - h - top
        left = (size - w) // 2
        right = size - w - left
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    warped_mask_cropped = pad_to_size(warped_mask_cropped, target_size)
    template_mask_cropped = pad_to_size(template_mask_cropped, target_size)

    # 3. Создаем новые src и dst, помещая кропы в центр черного квадрата
    src = np.float32(pad_to_size(warped_mask_cropped, target_size))
    dst = np.float32(pad_to_size(template_mask_cropped, target_size))
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
    max_radius = target_size * 0.80 


    # Флаг WARP_FILL_OUTLIERS заполняет "пустоты" нулями
    src_polar = cv2.linearPolar(np.float32(src), (cx_src, cy_src), max_radius, cv2.WARP_FILL_OUTLIERS)
    dst_polar = cv2.linearPolar(np.float32(dst), (cx_dst, cy_dst), max_radius, cv2.WARP_FILL_OUTLIERS)

    # 3. Фазовая корреляция находит сдвиг
    hann = cv2.createHanningWindow(src_polar.shape[:2][::-1], cv2.CV_32F)
    shift, _ = cv2.phaseCorrelate(src_polar, dst_polar, window=hann)

    # 6. Расчет угла
    dy = shift[1]
    angle = -(dy * 360.0) / src_polar.shape[0] # shape[0] это высота (h)
    angle = min(np.abs(90-np.abs(angle)), np.abs(angle))

    return float(angle)
