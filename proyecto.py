import cv2
import numpy as np
import math

# ---- Parámetros generales ----
MIRROR = False                 # Toggle espejo (tecla 'm')
preview_w, preview_h = 640, 480

# ---- Utilidades ----
def clamp(v, lo, hi):
    return int(max(lo, min(hi, v)))

def calibrate_from_roi(hsv_roi, p_low=5, p_high=95, margin_h=5, margin_sv=20):
    H = hsv_roi[:, :, 0].reshape(-1)
    S = hsv_roi[:, :, 1].reshape(-1)
    V = hsv_roi[:, :, 2].reshape(-1)
    mask_valid = (V > 10) & (V < 361) & (S > 20)
    if mask_valid.sum() > 200:
        H, S, V = H[mask_valid], S[mask_valid], V[mask_valid]
    h_lo, h_hi = np.percentile(H, [p_low, p_high]).astype(int)
    s_lo, s_hi = np.percentile(S, [p_low, p_high]).astype(int)
    v_lo, v_hi = np.percentile(V, [p_low, p_high]).astype(int)
    h_lo = clamp(h_lo - margin_h, 0, 179)
    h_hi = clamp(h_hi + margin_h, 0, 179)
    s_lo = clamp(s_lo - margin_sv, 0, 255)
    s_hi = clamp(s_hi + margin_sv, 0, 255)
    v_lo = clamp(v_lo - margin_sv, 0, 255)
    v_hi = clamp(v_hi + margin_sv, 0, 255)
    if h_lo > h_hi:
        h_lo, h_hi = 0, max(h_lo, h_hi)
    return np.array([h_lo, s_lo, v_lo], np.uint8), np.array([h_hi, s_hi, v_hi], np.uint8)

def clip_rect(x0, y0, x1, y1, w, h):
    x0, x1 = max(0, min(x0, w-1)), max(0, min(x1, w-1))
    y0, y1 = max(0, min(y0, h-1)), max(0, min(y1, h-1))
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0
    return x0, y0, x1, y1

def top_k_components(binary, k=2, min_area=1500):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    idxs = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    idxs = sorted(idxs, key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)[:k]
    comps = []
    for i in idxs:
        mask_i = np.where(labels == i, 255, 0).astype(np.uint8)
        x, y, w, h, area = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, cv2.CC_STAT_AREA]
        cx, cy = map(int, centroids[i])
        comps.append({"mask": mask_i, "bbox": (x, y, w, h), "centroid": (cx, cy), "area": area})
    return comps

def count_fingers_from_mask(mask, canvas):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, canvas
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 2000:
        return 0, canvas

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (255, 0, 0), 1)

    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return 0, canvas
    defects = cv2.convexityDefects(cnt, hull_idx)

    fingers = 0
    if defects is not None:
        valid = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            p1 = tuple(cnt[s][0]); p2 = tuple(cnt[e][0]); p3 = tuple(cnt[f][0])

            a = np.linalg.norm(np.array(p1) - np.array(p2))
            b = np.linalg.norm(np.array(p1) - np.array(p3))
            c = np.linalg.norm(np.array(p2) - np.array(p3))
            if b * c == 0:
                continue
            cosang = (b*b + c*c - a*a) / (2.0 * b * c)
            cosang = np.clip(cosang, -1.0, 1.0)
            angle = np.degrees(np.arccos(cosang))
            depth = d / 256.0

            if angle < 90 and depth > h * 0.04 and p3[1] < y + h * 0.85:
                valid += 1
                cv2.circle(canvas, p1, 4, (0, 255, 0), -1)
                cv2.circle(canvas, p2, 4, (0, 255, 0), -1)
                cv2.circle(canvas, p3, 4, (0, 0, 255), -1)
                cv2.line(canvas, p1, p3, (0, 255, 255), 1)
                cv2.line(canvas, p2, p3, (0, 255, 255), 1)

        fingers = min(valid + 1, 5) if valid > 0 else 0

    hull_pts = cv2.convexHull(cnt)
    cv2.drawContours(canvas, [cnt], -1, (255, 255, 0), 1)
    cv2.drawContours(canvas, [hull_pts], -1, (0, 255, 255), 1)
    return fingers, canvas

# ---- Entrada de vídeo ----
cap = cv2.VideoCapture(1)  # o 0 para webcam
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir el video.")

# ---- Estado interactivo ----
contador = 0
lower_skin, upper_skin = None, None
roi_coords = None
drag_start = None
drag_current = None
is_dragging = False

def on_mouse(event, x, y, flags, param):
    nonlocal_vars = param
    global roi_coords, drag_start, drag_current, is_dragging
    W, H = nonlocal_vars['w'], nonlocal_vars['h']
    if event == cv2.EVENT_LBUTTONDOWN:
        is_dragging = True
        drag_start = (x, y)
        drag_current = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        drag_current = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and is_dragging:
        is_dragging = False
        x0, y0, x1, y1 = clip_rect(drag_start[0], drag_start[1], x, y, W, H)
        if (x1 - x0) >= 5 and (y1 - y0) >= 5:
            roi_coords = (x0, y0, x1, y1)
        drag_start = None
        drag_current = None

cv2.namedWindow("Original")
mouse_params = {'w': preview_w, 'h': preview_h}
cv2.setMouseCallback("Original", on_mouse, mouse_params)

# ---- Bucle principal ----
while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video.")
        break

    frame = cv2.resize(frame, (preview_w, preview_h))
    if MIRROR:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    mouse_params['w'], mouse_params['h'] = w, h

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    vis = frame.copy()

    if lower_skin is not None and upper_skin is not None:
        raw_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.GaussianBlur(raw_mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        # separa manos pegadas por muñeca (ajusta iterations si hiciera falta)
        mask_sep = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

        components = top_k_components(mask_sep, k=2, min_area=1500)

        cv2.imshow("Mascara", mask_sep)
        skin_only = cv2.bitwise_and(frame, frame, mask=mask_sep)
        cv2.imshow("Solo piel (auto-calibrado)", skin_only)

        pairs = []  # [(cx, dedos)]
        for comp in components:
            dedos, vis = count_fingers_from_mask(comp["mask"], vis)
            x0, y0, ww, hh = comp["bbox"]
            cx, cy = comp["centroid"]
            pairs.append((cx, dedos))
            cv2.rectangle(vis, (x0, y0), (x0+ww, y0+hh), (0, 200, 255), 2)
            cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)
            cv2.putText(vis, f"{dedos}", (x0, max(20, y0-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)

        pairs.sort(key=lambda t: t[0])
        if len(pairs) == 2:
            izq, der = pairs[0][1], pairs[1][1]
            cv2.putText(vis, f"Izq:{izq}  Der:{der}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        elif len(pairs) == 1:
            cv2.putText(vis, f"Una mano: {pairs[0][1]}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        else:
            cv2.putText(vis, "Sin manos", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    else:
        cv2.imshow("Solo piel (auto-calibrado)", np.zeros_like(frame))
        cv2.imshow("Mascara", np.zeros((h, w), dtype=np.uint8))

    # ROI visual
    if drag_start is not None and drag_current is not None:
        x0, y0, x1, y1 = clip_rect(drag_start[0], drag_start[1], drag_current[0], drag_current[1], w, h)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
    if roi_coords is not None:
        x0, y0, x1, y1 = roi_coords
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 0), 2)

    # HUD
    hud = "c: calibrar | r: reset | s: guardar | m: espejo | q: salir"
    cv2.putText(vis, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if lower_skin is not None:
        cv2.putText(vis, f"HSV lower:{tuple(int(v) for v in lower_skin)} upper:{tuple(int(v) for v in upper_skin)}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)

    cv2.imshow("Original", vis)

    # Teclado
    key = cv2.waitKey(20) & 0xFF
    if key == ord('c'):
        if roi_coords is None:
            print("Selecciona un ROI arrastrando con el raton (se usara uno centrado por defecto).")
            cx, cy = w // 2, h // 2
            x0, y0, x1, y1 = clip_rect(cx - 50, cy - 50, cx + 50, cy + 50, w, h)
        else:
            x0, y0, x1, y1 = roi_coords
        roi = hsv[y0:y1, x0:x1]
        lower_skin, upper_skin = calibrate_from_roi(roi)
        print("Calibrado HSV:")
        print(" lower:", lower_skin, " upper:", upper_skin)
    elif key == ord('r'):
        lower_skin, upper_skin = None, None
        roi_coords = None
        print("Calibracion reseteada.")
    elif key == ord('s') and lower_skin is not None:
        # guarda imagen filtrada por la máscara actual
        skin_only = cv2.bitwise_and(frame, frame, mask=(cv2.inRange(hsv, lower_skin, upper_skin)))
        nombre = f"piel_{contador}.png"
        cv2.imwrite(nombre, skin_only)
        print(f"Imagen guardada: {nombre}")
        contador += 1
    elif key == ord('m'):
        MIRROR = not MIRROR
        print("Espejo:", "ON" if MIRROR else "OFF")
    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
