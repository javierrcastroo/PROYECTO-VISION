import cv2
import numpy as np
from collections import deque, Counter

from config import (
    PREVIEW_W, PREVIEW_H,
    RECOGNIZE_MODE,
    NORMALIZED_SIZE,
    CONFIDENCE_THRESHOLD,
)   
from segmentation import calibrate_from_roi, segment_hand_mask
from features import compute_feature_vector
from storage import save_gesture_example, load_gesture_gallery, save_sequence_json
from classifier import knn_predict
import ui  # importamos el módulo completo para acceder a su estado global
import os
from config import CAMERA_PARAMS_PATH, USE_UNDISTORT


def majority_vote(labels):
    """
    Devuelve la moda de una lista de labels
    (ej. ['2dedos','2dedos','????',...]).
    Si la lista está vacía, devuelve None.
    """
    if not labels:
        return None
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def main():
    # Abrir cámara
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    cv2.namedWindow("Original")
    cv2.setMouseCallback("Original", ui.mouse_callback)

    # Estado dinámico
    lower_skin, upper_skin = None, None
    gallery = load_gesture_gallery() if RECOGNIZE_MODE else []

    contador_guardadas = 0
    acciones = []  # Secuencia de gestos confirmada con 'a'

    # Etiqueta activa en runtime (para guardar con 'g')
    current_label = "2dedos"

    # Cola de predicciones recientes para suavizado temporal
    # Guardamos las etiquetas ya filtradas por confianza (o "????")
    recent_preds = deque(maxlen=7)

    # intentar cargar parámetros de cámara (para undistort)
    if USE_UNDISTORT and os.path.exists(CAMERA_PARAMS_PATH):
        _params = np.load(CAMERA_PARAMS_PATH)
        CAM_MTX = _params["camera_matrix"]
        DIST_COEFFS = _params["dist_coeffs"]
        print("[INFO] Parámetros de cámara cargados, se aplicará undistort.")
    else:
        CAM_MTX = None
        DIST_COEFFS = None
        if USE_UNDISTORT:
            print("[WARN] USE_UNDISTORT=True pero no existe camera_params.npz")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if CAM_MTX is not None:
            frame = cv2.undistort(frame, CAM_MTX, DIST_COEFFS)
        # modo espejo (comenta esta línea si no quieres espejo)
        frame = cv2.flip(frame, 1)

        frame = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        vis = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. Dibujar ROI actual en pantalla (rectángulo que marcas con el ratón)
        ui.draw_roi_rectangle(vis)

        # 2. Segmentar la mano (máscara limpia 0/255)
        mask_largest = segment_hand_mask(hsv, lower_skin, upper_skin)

        # 3. Dibuja caja mínima que encierra la mano (minAreaRect) sobre la vista
        ui.draw_hand_box(vis, mask_largest)

        # 4. Para debug visual: mano coloreada
        skin_only = cv2.bitwise_and(frame, frame, mask=mask_largest)

        # 5. Extraer features y predecir con k-NN
        best_dist = None
        per_frame_label = None  # etiqueta para ESTE frame tras filtrado de confianza

        feature_vec = compute_feature_vector(mask_largest)
        if feature_vec is not None and RECOGNIZE_MODE:
            raw_label, best_dist = knn_predict(feature_vec, gallery, k=5)

            # Rechazo por baja confianza:
            if raw_label is not None and best_dist is not None:
                if best_dist > CONFIDENCE_THRESHOLD:
                    per_frame_label = "????"
                else:
                    per_frame_label = raw_label

        # 6. Guardar predicción filtrada en la cola temporal
        if per_frame_label is not None:
            recent_preds.append(per_frame_label)

        # 7. Suavizado temporal: moda de las últimas N predicciones
        stable_label = majority_vote(list(recent_preds))

        # 8. HUD: instrucciones + etiqueta activa actual
        ui.draw_hud(vis, lower_skin, upper_skin, current_label)

        # 9. Pintar la etiqueta suavizada en pantalla
        ui.draw_prediction(
            vis,
            stable_label,
            best_dist if best_dist is not None else 0.0
        )

        # 10. Mostrar ventanas
        cv2.imshow("Original", vis)
        cv2.imshow("Mascara", mask_largest)
        cv2.imshow("Solo piel", skin_only)

        # 11. Leer teclado
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Calibrar HSV desde el ROI marcado
            if ui.roi_defined:
                x0, x1 = sorted([ui.x_start, ui.x_end])
                y0, y1 = sorted([ui.y_start, ui.y_end])
                if (x1 - x0) > 5 and (y1 - y0) > 5:
                    roi_hsv = hsv[y0:y1, x0:x1]
                    lower_skin, upper_skin = calibrate_from_roi(
                        roi_hsv,
                        p_low=5, p_high=95,
                        margin_h=5, margin_sv=20
                    )
                    print("[INFO] Calibrado HSV:")
                    print(" lower:", lower_skin, " upper:", upper_skin)
                else:
                    print("[WARN] ROI demasiado pequeño para calibrar.")
            else:
                print("[WARN] Primero dibuja un ROI con el ratón.")

        elif key == ord('r'):
            # Reset calibración
            lower_skin, upper_skin = None, None
            print("[INFO] Calibración reseteada.")

        elif key == ord('g'):
            # Guardar ejemplo actual en disco con la etiqueta activa
            if lower_skin is None or upper_skin is None:
                print("[WARN] No hay calibración HSV. No guardo.")
            else:
                feat_vec = compute_feature_vector(mask_largest)
                if feat_vec is None:
                    print("[WARN] No se detecta mano válida para guardar.")
                else:
                    save_gesture_example(feat_vec, current_label)
                    contador_guardadas += 1
                    # Actualizar galería en caliente, para que el clasificador mejore al vuelo
                    if RECOGNIZE_MODE:
                        gallery.append((feat_vec, current_label))
                    print(f"[INFO] Guardadas: {contador_guardadas} (label={current_label})")

        elif key == ord('a'):
            # Añadir el gesto estable (ya suavizado) a la secuencia
            acciones = ui.append_action(acciones, stable_label)

        elif key == ord('p'):
            # 1) Guardar la secuencia en JSON con timestamp
            save_sequence_json(acciones)

            # 2) Mostrar la secuencia en consola
            print(f"[INFO] Lista de gestos capturada: {acciones}")

            # 3) Vaciar la secuencia
            acciones.clear()
            print("[INFO] Lista de gestos reiniciada.")

        # Cambiar etiqueta activa en caliente (para guardar dataset con 'g')
        elif key == ord('0'):
            current_label = "0dedos"
            print(f"[INFO] current_label -> {current_label}")
        elif key == ord('1'):
            current_label = "1dedo"
            print(f"[INFO] current_label -> {current_label}")
        elif key == ord('2'):
            current_label = "2dedos"
            print(f"[INFO] current_label -> {current_label}")
        elif key == ord('3'):
            current_label = "3dedos"
            print(f"[INFO] current_label -> {current_label}")
        elif key == ord('4'):
            current_label = "4dedos"
            print(f"[INFO] current_label -> {current_label}")
        elif key == ord('5'):
            current_label = "5dedos"
            print(f"[INFO] current_label -> {current_label}")

        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
