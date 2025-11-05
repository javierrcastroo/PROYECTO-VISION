# object_tracker.py
import cv2
import numpy as np

# color por defecto de objetos (luego lo calibras con 'o')
OBJ_LOWER_DEFAULT = np.array([0, 120, 80], dtype=np.uint8)
OBJ_UPPER_DEFAULT = np.array([15, 255, 255], dtype=np.uint8)

# color por defecto del origen (luego lo calibras con 'r')
ORIG_LOWER_DEFAULT = np.array([90, 120, 80], dtype=np.uint8)   # azulito por defecto
ORIG_UPPER_DEFAULT = np.array([130, 255, 255], dtype=np.uint8)

current_obj_lower = OBJ_LOWER_DEFAULT.copy()
current_obj_upper = OBJ_UPPER_DEFAULT.copy()

current_origin_lower = ORIG_LOWER_DEFAULT.copy()
current_origin_upper = ORIG_UPPER_DEFAULT.copy()


def _calibrate_from_roi(hsv_roi, p_low=5, p_high=95, margin_h=3, margin_sv=20):
    H = hsv_roi[:,:,0].reshape(-1)
    S = hsv_roi[:,:,1].reshape(-1)
    V = hsv_roi[:,:,2].reshape(-1)

    mask_valid = (V > 40) & (S > 20)
    if mask_valid.sum() > 200:
        H, S, V = H[mask_valid], S[mask_valid], V[mask_valid]

    h_lo, h_hi = np.percentile(H, [p_low, p_high]).astype(int)
    s_lo, s_hi = np.percentile(S, [p_low, p_high]).astype(int)
    v_lo, v_hi = np.percentile(V, [p_low, p_high]).astype(int)

    h_lo = max(0, h_lo - margin_h)
    h_hi = min(179, h_hi + margin_h)
    s_lo = max(0, s_lo - margin_sv)
    s_hi = min(255, s_hi + margin_sv)
    v_lo = max(0, v_lo - margin_sv)
    v_hi = min(255, v_hi + margin_sv)

    lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
    return lower, upper


def calibrate_object_color_from_roi(hsv_roi):
    return _calibrate_from_roi(hsv_roi)


def calibrate_origin_color_from_roi(hsv_roi):
    return _calibrate_from_roi(hsv_roi)


def detect_colored_points_in_board(hsv_frame, board_quad, lower, upper, max_objs=4, min_area=50):
    """
    hsv_frame: frame del tablero en HSV
    board_quad: 4x2 float32 (tl,tr,br,bl)
    lower/upper: rango HSV del objeto/origen
    Devuelve:
      centers: lista de (x,y) en coordenadas de imagen
      mask: máscara de ese color (para debug)
    """
    mask = cv2.inRange(hsv_frame, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], mask

    # polígono del tablero para filtrar puntos fuera
    board_poly = np.array(board_quad, dtype=np.float32)

    # ordenar contornos por área desc
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    centers = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # comprobar si está dentro del tablero
        if cv2.pointPolygonTest(board_poly, (cx, cy), False) >= 0:
            centers.append((cx, cy))

        if len(centers) >= max_objs:
            break

    return centers, mask
