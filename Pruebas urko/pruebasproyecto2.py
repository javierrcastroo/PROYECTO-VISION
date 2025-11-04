import cv2
import numpy as np
import os
import glob
import time

# =========================
# CONFIGURACIÓN BÁSICA
# =========================

SAVE_DIR = "gestures2"
os.makedirs(SAVE_DIR, exist_ok=True)

CURRENT_LABEL = "ok"        # Nombre del gesto actual
RECOGNIZE_MODE = True           # Si está activo, intenta reconocer en vivo

preview_w, preview_h = 640, 480

# Kernels para operaciones morfológicas (eliminan ruido)
KERNEL_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

# =========================
# FUNCIONES AUXILIARES
# =========================

def clamp(v, lo, hi):
    """Limita un valor entre dos límites."""
    return int(max(lo, min(hi, v)))

def calibrate_from_roi(hsv_roi, p_low=2, p_high=97, margin_h=2, margin_sv=10):
    """Calcula los límites HSV para la piel usando un ROI dibujado."""
    H = hsv_roi[:, :, 0].reshape(-1)
    S = hsv_roi[:, :, 1].reshape(-1)
    V = hsv_roi[:, :, 2].reshape(-1)

    mask_valid = (V > 30) & (V < 245) & (S > 20)
    if mask_valid.sum() > 200:
        H, S, V = H[mask_valid], S[mask_valid], V[mask_valid]

    h_lo, h_hi = np.percentile(H, [p_low, p_high]).astype(int)
    s_lo, s_hi = np.percentile(S, [p_low, p_high]).astype(int)
    v_lo, v_hi = np.percentile(V, [p_low, p_high]).astype(int)

    # Se amplían los márgenes para tolerancia de iluminación
    h_lo = clamp(h_lo - margin_h, 0, 179)
    h_hi = clamp(h_hi + margin_h, 0, 179)
    s_lo = clamp(s_lo - margin_sv, 0, 255)
    s_hi = clamp(s_hi + margin_sv, 0, 255)
    v_lo = clamp(v_lo - margin_sv, 0, 255)
    v_hi = clamp(v_hi + margin_sv, 0, 255)

    lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
    return lower, upper

def largest_component_mask(binary):
    """Devuelve solo la componente blanca más grande (mano)."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)

# =========================
# CÁLCULO DE FEATURES
# =========================

def compute_hu_moments(contour):
    """Momentos de Hu (7 valores invariantes)."""
    M = cv2.moments(contour)
    hu = cv2.HuMoments(M).flatten()
    return hu

def compute_radial_signature(mask, contour, num_angles=36):
    """Firma radial: mide la distancia desde el centroide hasta el borde."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return np.zeros(num_angles, dtype=np.float32)

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    h, w = mask.shape[:2]
    max_distances = []
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)

    for theta in angles:
        dx = np.cos(theta)
        dy = np.sin(theta)
        dist = 0.0
        step = 1.0
        while True:
            x = int(round(cx + dist*dx))
            y = int(round(cy + dist*dy))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            if mask[y, x] == 0:
                break
            dist += step
        max_distances.append(dist)

    # Normaliza distancias (0–1)
    max_distances = np.array(max_distances, dtype=np.float32)
    m = max_distances.max() if max_distances.max() > 0 else 1.0
    radial_norm = max_distances / m
    return radial_norm

def compute_feature_vector(mask):
    """Devuelve el vector de características + bbox + contorno."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:
        return None, None, None

    hu = compute_hu_moments(c)
    radial = compute_radial_signature(mask, c, 36)
    feat = np.concatenate([hu, radial.astype(np.float64)], axis=0)

    x, y, w, h = cv2.boundingRect(c)
    return feat, (x, y, w, h), c

# =========================
# GUARDAR Y CARGAR GESTOS
# =========================

def save_gesture_example(mask, feature_vec, label, save_dir=SAVE_DIR):
    """Guarda máscara + vector + etiqueta como archivo .npz."""
    ts = int(time.time() * 1000)
    filepath = os.path.join(save_dir, f"{label}_{ts}.npz")
    np.savez(filepath, mask=mask, feature=feature_vec, label=label)
    print(f"[INFO] Gesto guardado en {filepath}")

def load_gesture_gallery(save_dir=SAVE_DIR):
    """Carga todos los ejemplos guardados en memoria."""
    gallery = []
    for fp in glob.glob(os.path.join(save_dir, "*.npz")):
        data = np.load(fp, allow_pickle=True)
        feature = data["feature"]
        label = str(data["label"])
        gallery.append((feature, label))
    print(f"[INFO] Cargadas {len(gallery)} muestras en galería.")
    return gallery

def nearest_neighbor_predict(feature_vec, gallery):
    """Busca el gesto más parecido por distancia euclidiana."""
    if len(gallery) == 0:
        return None, None
    best_label, best_dist = None, None
    for (f_ref, lab) in gallery:
        if f_ref.shape != feature_vec.shape:
            continue
        dist = np.linalg.norm(feature_vec - f_ref)
        if (best_dist is None) or (dist < best_dist):
            best_dist = dist
            best_label = lab
    return best_label, best_dist

# =========================
# SELECCIÓN DE ROI CON RATÓN
# =========================

roi_selecting = False
roi_defined   = False
x_start = y_start = x_end = y_end = 0

def mouse_callback(event, x, y, flags, param):
    """Permite seleccionar un ROI para calibrar el color de piel."""
    global roi_selecting, roi_defined, x_start, y_start, x_end, y_end
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_selecting = True
        roi_defined   = False
        x_start, y_start = x, y
        x_end,   y_end   = x, y
    elif event == cv2.EVENT_MOUSEMOVE and roi_selecting:
        x_end, y_end = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        roi_selecting = False
        roi_defined   = True
        x_end, y_end  = x, y

# =========================
# PROGRAMA PRINCIPAL
# =========================

def main():
    global roi_defined, x_start, y_start, x_end, y_end

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    cv2.namedWindow("Original")
    cv2.setMouseCallback("Original", mouse_callback)

    lower_skin, upper_skin = None, None
    gallery = load_gesture_gallery(SAVE_DIR) if RECOGNIZE_MODE else []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (preview_w, preview_h))
        vis = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Dibuja el rectángulo de selección (ROI)
        if roi_selecting or roi_defined:
            cv2.rectangle(vis, (x_start, y_start), (x_end, y_end), (0, 255, 255), 2)

        mask_largest = np.zeros(frame.shape[:2], dtype=np.uint8)
        skin_only = np.zeros_like(frame)

        if lower_skin is not None and upper_skin is not None:
            # Segmenta piel según calibración HSV
            raw_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            mask = cv2.GaussianBlur(raw_mask, (5, 5), 0)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL_OPEN)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_CLOSE)
            mask_largest = largest_component_mask(mask)
            skin_only = cv2.bitwise_and(frame, frame, mask=mask_largest)

            predicted_label = None
            best_dist = None
            feature_vec, bbox, contour = compute_feature_vector(mask_largest)

            # 🔸 Dibuja también el rectángulo mínimo en "Solo piel"
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(skin_only, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Si está activo el modo reconocimiento, predice el gesto
            if feature_vec is not None and RECOGNIZE_MODE:
                predicted_label, best_dist = nearest_neighbor_predict(feature_vec, gallery)

            # Muestra el resultado del reconocimiento
            if predicted_label is not None:
                cv2.putText(vis,
                            f"Gesto: {predicted_label} (dist={best_dist:.2f})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis,
                            f"Gesto: {predicted_label} (dist={best_dist:.2f})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 1, cv2.LINE_AA)

        # HUD con instrucciones
        hud = "ROI: click y arrastra | 'c' calibrar | 'g' guardar | 'r' reset | 'q' salir"
        cv2.putText(vis, hud, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Muestra info HSV actual
        if lower_skin is not None:
            cv2.putText(vis,
                        f"HSV low:{tuple(int(v) for v in lower_skin)} up:{tuple(int(v) for v in upper_skin)}",
                        (10, preview_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (180,255,180), 1, cv2.LINE_AA)

        # Ventanas principales
        cv2.imshow("Original", vis)
        cv2.imshow("Mascara", mask_largest)
        cv2.imshow("Solo piel", skin_only)

        key = cv2.waitKey(1) & 0xFF

        # ======= Controles de teclado =======

        if key == ord('c'):
            # Calibrar color de piel desde ROI
            if roi_defined:
                x0, x1 = sorted([x_start, x_end])
                y0, y1 = sorted([y_start, y_end])
                if (x1 - x0) > 5 and (y1 - y0) > 5:
                    roi_hsv = hsv[y0:y1, x0:x1]
                    lower_skin, upper_skin = calibrate_from_roi(roi_hsv,
                                                                p_low=5, p_high=95,
                                                                margin_h=5, margin_sv=20)
                    print("[INFO] Calibrado HSV:", lower_skin, upper_skin)
                else:
                    print("[WARN] ROI demasiado pequeño.")
            else:
                print("[WARN] Primero dibuja un ROI.")

        elif key == ord('r'):
            lower_skin, upper_skin = None, None
            print("[INFO] Calibración reseteada.")

        elif key == ord('g'):
            # Guarda gesto actual (solo recorte de la mano)
            if lower_skin is None or upper_skin is None:
                print("[WARN] No hay calibración HSV. No guardo.")
            else:
                feat_vec, bbox, contour = compute_feature_vector(mask_largest)
                if feat_vec is None:
                    print("[WARN] No se detecta mano válida.")
                else:
                    x, y, w, h = bbox
                    hand_crop = mask_largest[y:y+h, x:x+w]
                    hand_crop = cv2.resize(hand_crop, (200, 200))  # Normaliza tamaño
                    save_gesture_example(hand_crop, feat_vec, CURRENT_LABEL, SAVE_DIR)
                    if RECOGNIZE_MODE:
                        gallery.append((feat_vec, CURRENT_LABEL))

        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
