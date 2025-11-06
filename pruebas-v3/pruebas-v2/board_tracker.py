# board_tracker.py
import cv2
import numpy as np

# nº de casillas que tiene tu tablero
BOARD_SQUARES = 5
# tamaño real de una casilla (cm)
SQUARE_SIZE_CM = 3.7

# rango por defecto por si aún no has calibrado
DEFAULT_LOWER = np.array([5, 80, 80], dtype=np.uint8)
DEFAULT_UPPER = np.array([25, 255, 255], dtype=np.uint8)

# rangos que realmente usamos (se pueden cambiar con 'b')
current_lower = DEFAULT_LOWER.copy()
current_upper = DEFAULT_UPPER.copy()


def calibrate_board_color_from_roi(hsv_roi, p_low=5, p_high=95, margin_h=3, margin_sv=20):
    """
    Calcula un rango HSV (lower, upper) a partir de un ROI del tablero.
    Esto es lo que llama tu main cuando pulsas 'b'.
    """
    H = hsv_roi[:, :, 0].reshape(-1)
    S = hsv_roi[:, :, 1].reshape(-1)
    V = hsv_roi[:, :, 2].reshape(-1)

    # filtrar blancos / cosas poco saturadas
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


def detect_board(frame, camera_matrix=None, dist_coeffs=None):
    """
    Detecta el tablero por color (naranja), une las casillas con morfología gorda,
    saca el rectángulo exterior y dibuja una rejilla 5x5 dentro.

    Devuelve:
      vis            -> frame con el tablero dibujado
      found          -> True/False
      ratio_cm_per_pix
      height_px
      mask_show      -> máscara SIN el close gordo (para que la enseñes en otra ventana)
      quad           -> las 4 esquinas del tablero (tl,tr,br,bl) o None
    """
    global current_lower, current_upper

    # undistort si tenemos params
    if camera_matrix is not None and dist_coeffs is not None:
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    vis = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # máscara básica de color
    mask = cv2.inRange(hsv, current_lower, current_upper)
    mask_show = mask.copy()

    # unir todas las casillas naranjas separadas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask_merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cv2.putText(vis, "Buscando tablero...", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return vis, False, None, None, mask_show, None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 5000:
        cv2.putText(vis, "Tablero muy pequeño...", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return vis, False, None, None, mask_show, None

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.polylines(vis, [box], True, (0, 255, 255), 2)

    quad = order_points(box.astype(np.float32))

    # dibujar rejilla dentro
    draw_grid_in_quad(vis, quad, BOARD_SQUARES)

    # medir alto en píxeles
    tl, tr, br, bl = quad
    top_mid = (tl + tr) / 2.0
    bot_mid = (bl + br) / 2.0
    height_px = float(np.linalg.norm(top_mid - bot_mid))

    ratio_cm_per_pix = None
    if height_px > 1e-3:
        pix_per_square = height_px / BOARD_SQUARES
        ratio_cm_per_pix = SQUARE_SIZE_CM / pix_per_square

    if ratio_cm_per_pix is not None:
        est_height_cm = height_px * ratio_cm_per_pix
        cv2.putText(
            vis,
            f"OK | {ratio_cm_per_pix:.4f} cm/pix | alto={est_height_cm:.1f}cm",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 0), 2
        )
    else:
        cv2.putText(vis, "Tablero OK", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    return vis, True, ratio_cm_per_pix, height_px, mask_show, quad


def order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def draw_grid_in_quad(img, quad, n=5):
    quad = quad.astype(np.float32)
    tl, tr, br, bl = quad

    # horizontales
    for i in range(n + 1):
        t = i / n
        p1 = tl * (1 - t) + bl * t
        p2 = tr * (1 - t) + br * t
        cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 0, 0), 1)

    # verticales
    for j in range(n + 1):
        t = j / n
        p1 = tl * (1 - t) + tr * t
        p2 = bl * (1 - t) + br * t
        cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 255, 0), 1)
