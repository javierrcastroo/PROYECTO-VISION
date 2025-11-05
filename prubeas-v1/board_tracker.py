# board_tracker.py (solo la función detect_board actualizada)
import cv2
import numpy as np

BOARD_SQUARES = 5
SQUARE_SIZE_CM = 3.7

DEFAULT_LOWER = np.array([5, 80, 80], dtype=np.uint8)
DEFAULT_UPPER = np.array([25, 255, 255], dtype=np.uint8)
current_lower = DEFAULT_LOWER.copy()
current_upper = DEFAULT_UPPER.copy()

def detect_board(frame, camera_matrix=None, dist_coeffs=None):
    global current_lower, current_upper

    if camera_matrix is not None and dist_coeffs is not None:
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    vis = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, current_lower, current_upper)
    mask_show = mask.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask_merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cv2.putText(vis, "Buscando tablero...", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        return vis, False, None, None, mask_show, None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 5000:
        cv2.putText(vis, "Tablero pequeño...", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        return vis, False, None, None, mask_show, None

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.polylines(vis, [box], True, (0,255,255), 2)

    quad = order_points(box.astype(np.float32))
    draw_grid_in_quad(vis, quad, BOARD_SQUARES)

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
        cv2.putText(vis,
                    f"OK | {ratio_cm_per_pix:.4f} cm/pix | alto={est_height_cm:.1f}cm",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
    else:
        cv2.putText(vis, "Tablero OK", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

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
    for i in range(n+1):
        t = i / n
        p1 = tl*(1-t) + bl*t
        p2 = tr*(1-t) + br*t
        cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (255,0,0), 1)
    for j in range(n+1):
        t = j / n
        p1 = tl*(1-t) + tr*t
        p2 = bl*(1-t) + br*t
        cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (0,255,0), 1)
