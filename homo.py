import cv2
import numpy as np
from pathlib import Path


def mask_preprocessing(mask: np.ndarray) -> np.ndarray:
    m = mask.copy()

    # Приводим к grayscale, если нужно
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    elif m.ndim != 2:
        raise ValueError(f"Unsupported mask ndim: {m.ndim}")

    # Гарантируем uint8
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)

    # Если маска бинарная — нормализуем к 0/255
    unique_vals = np.unique(m)
    if len(unique_vals) <= 2:
        m = np.where(m > 0, 255, 0).astype(np.uint8)

    # --- Морфология ---
    k = max(3, int(round(min(m.shape[:2]) / 100)))
    kernel = np.ones((k, k), np.uint8)

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Контурное сглаживание
    m = cv2.dilate(m, kernel, iterations=1)
    m = cv2.erode(m, kernel, iterations=1)

    m = (m > 0).astype(np.uint8)

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
    


def get_max_radius(mask, cx, cy):
    y_coords, x_coords = np.nonzero(mask)  # быстрее, возвращает int arrays
    if x_coords.size == 0:
        return 0
    d2 = (x_coords - cx).astype(np.int64)**2 + (y_coords - cy).astype(np.int64)**2
    return int(np.sqrt(d2.max()))



def normalize_masks(mask1, mask2):
    # Находим центры масс и площади

    x_1, y_1, w_c_1, h_c_1 = cv2.boundingRect(cv2.findNonZero(mask1))
    mask1 = mask1[y_1:y_1+h_c_1, x_1:x_1+w_c_1]

    x_2, y_2, w_c_2, h_c_2 = cv2.boundingRect(cv2.findNonZero(mask2))
    mask2 = mask2[y_2:y_2+h_c_2, x_2:x_2+w_c_2]

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
    # Определяем размер выходного изображения

    radius = max(get_max_radius(mask1, c1[0], c1[1]),
             get_max_radius(mask2, c2[0], c2[1]))
    max_side = max(mask1.shape[0], mask2.shape[0], mask2.shape[0], mask2.shape[1])
    target_size = max(int(np.ceil(np.hypot(max_side, max_side))), 2*radius)

    if target_size % 2 != 0: target_size += 51
    else: target_size += 52

    # Центрируем обе маски
    canvas1 = np.zeros((target_size, target_size), dtype=np.uint8)
    canvas2 = np.zeros((target_size, target_size), dtype=np.uint8)

    #print(canvas1.shape, mask1.shape,mask2.shape )
    #cv2.imwrite("img/mask1.png", mask1*255)
    #cv2.imwrite("img/mask2.png", mask2*255)
    
    y1, x1 = target_size // 2 - c1[1], target_size // 2 - c1[0]
    y2, x2 = target_size // 2 - c2[1], target_size // 2 - c2[0]

    canvas1[y1:y1+mask1.shape[0], x1:x1+mask1.shape[1]] = mask1
    canvas2[y2:y2+mask2.shape[0], x2:x2+mask2.shape[1]] = mask2

    return canvas1, canvas2

def rotation(mask: np.ndarray, angle):
    m = cv2.moments(mask)
    rotated = cv2.warpAffine(mask, cv2.getRotationMatrix2D((int(m['m10']/m['m00']), int(m['m01']/m['m00'])), angle, 1.0), (mask.shape[1], mask.shape[0]))
    return rotated

def iou(a, b):
    # Ожидаем uint8 бинарные маски {0,1} или {0,255}
    if a.dtype != np.uint8:
        a = (a > 0).astype(np.uint8)
    if b.dtype != np.uint8:
        b = (b > 0).astype(np.uint8)

    # Привести к 0/255 (cv2.bitwise_and работает быстрее для uint8)
    if a.max() == 1:
        a = a * 255
    if b.max() == 1:
        b = b * 255

    inter = cv2.bitwise_and(a, b)
    union = cv2.bitwise_or(a, b)
    inter_count = cv2.countNonZero(inter)
    union_count = cv2.countNonZero(union)
    return (inter_count / union_count) if union_count > 0 else 0.0

def precompute_rotations(mask: np.ndarray, angles=range(-30, 31)):
    mask = (mask > 0).astype(np.uint8)

    # === ТВОЙ КОД: центр через moments (НЕ ТРОГАЕМ) ===
    m = cv2.moments(mask)
    if m['m00'] == 0:
        cx = mask.shape[1] // 2
        cy = mask.shape[0] // 2
    else:
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])

    h, w = mask.shape[:2]

    # === ДОБАВЛЯЕМ PADDING (МИНИМАЛЬНО) ===
    base = max(h, w)
    target_size = int(np.ceil(np.hypot(base, base)))
    if target_size % 2 != 0:  target_size += 51
    else:  target_size += 52

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)

    # сдвиг, чтобы центр маски (cx,cy) оказался в центре canvas
    y0 = target_size // 2 - cy
    x0 = target_size // 2 - cx

    canvas[y0:y0+h, x0:x0+w] = mask

    # новый центр — тот же самый объектный центр, но уже в canvas
    cx_p = target_size // 2
    cy_p = target_size // 2

    rotations = {}
    for angle in angles:
        M = cv2.getRotationMatrix2D((cx_p, cy_p), angle, 1.0)
        rotated = cv2.warpAffine( canvas,  M,  (target_size, target_size),  flags=cv2.INTER_NEAREST, borderValue=0 )
        rotations[angle] = rotated

    return rotations

def crop_to_union(a: np.ndarray, b: np.ndarray):
    # a и b — бинарные uint8 маски (0/255 или 0/1)
    # приводим к 0/255 для findNonZero
    aa = (a > 0).astype(np.uint8) * 255
    bb = (b > 0).astype(np.uint8) * 255
    nonz = cv2.bitwise_or(aa, bb)
    pts = cv2.findNonZero(nonz)
    if pts is None:
        return aa, bb  # обе пустые
    x, y, w, h = cv2.boundingRect(pts)
    return aa[y:y+h, x:x+w], bb[y:y+h, x:x+w]


def calculate_angle_line_polar(warped_mask: np.ndarray, template_mask: np.ndarray) -> float:

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
    #angle = min(np.abs(90-np.abs(angle)), np.abs(angle))
    angle = np.abs(angle)

    return float(angle)

def calculate_angle_iou(warped_mask: np.ndarray, template_mask: np.ndarray) -> float:
    best_angle = None
    best_score = -1

    for angle in range(-30, 31):
        rotated = rotation(warped_mask, angle)
        score = iou(rotated, template_mask)
        if score > best_score:
            best_score = score
            best_angle = angle
    return float(best_angle), best_score

def calculate_angle_iou_precomputed_iou(rotated_masks: dict, template_mask: np.ndarray):
    best_angle = None
    best_score = -1
    for angle, rotated in rotated_masks.items():
        score = iou(rotated, template_mask)
        if score > best_score:
            best_score = score
            best_angle = angle
    return float(best_angle), best_score


def find_template(mask: np.ndarray):

    best_template_mask = None
    best_angle = None
    best_score = -1

    x, y, w_c, h_c = cv2.boundingRect(cv2.findNonZero(mask))
    mask = mask[y:y+h_c, x:x+w_c]

    rotations = precompute_rotations(mask, angles=range(-30, 31))

    #cv2.imwrite("img/rotations_30.png", rotations[30]*255)

    for p in Path("template_masks").glob("*.*"):
        if p.suffix.lower() not in (".png",".jpg",".jpeg",".tif",".tiff",".bmp"): continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        template_mask = (img[:,:,0] > 0).astype(np.uint8)         # булева маска


        x2, y2, w_c2, h_c2 = cv2.boundingRect(cv2.findNonZero(template_mask))
        template_mask = template_mask[y2:y2+h_c2, x2:x2+w_c2]

        max_dim = max(512, max(mask.shape))
        scale = max(1.0, max(template_mask.shape) / max_dim)
        if scale > 1.0:
            template_mask = cv2.resize(template_mask, (int(template_mask.shape[1]/scale), int(template_mask.shape[0]/scale)), interpolation=cv2.INTER_NEAREST)

        mask, template_mask = normalize_masks(mask, template_mask)
        for flip in (None, 0, 1, -1):  # orig, vert, horiz, both
            template_mask = template_mask if flip is None else cv2.flip(template_mask, flip)
            for template_angle in (0, 90):
                template_mask = rotation(template_mask, template_angle)
                for angle, rotated in rotations.items():
                    rotated, template_mask = normalize_masks(rotated, template_mask)
                    a_crop, b_crop = crop_to_union(rotated, template_mask)
                    score = iou(a_crop, b_crop)
                    if score > best_score:
                        best_score = score
                        best_angle = angle
                        best_template_mask = template_mask
                        print(best_score, angle)
                        cv2.imwrite("img/best_template_mask.png", best_template_mask*255)
                        cv2.imwrite("img/rotated.png", rotated*255)


    return float(best_angle), best_template_mask



def detect_sheet_angle_no_homography(warped_mask: np.ndarray) -> float:



    angle, template_mask = find_template(warped_mask)

    #norm_warped_mask, norm_template_mask = normalize_masks(warped_mask_cropped, template_mask_cropped)

    #norm_warped_mask = rotation(norm_warped_mask, 20)

    #cv2.imwrite("img/norm_warped_mask_rot.png", norm_warped_mask*255)
    #norm_warped_mask = mask_preprocessing(norm_warped_mask)
    #norm_warped_mask, norm_template_mask = normalize_masks(norm_warped_mask, norm_template_mask)

    #calculate_angle_line_polar(norm_warped_mask, norm_template_mask)

    return angle
