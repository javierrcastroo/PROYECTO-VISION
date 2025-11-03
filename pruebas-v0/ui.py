import cv2
import numpy as np
from config import LABEL_TO_NUM, PREVIEW_H

# === Estado global del ROI ===
roi_selecting = False
roi_defined   = False
x_start = y_start = x_end = y_end = 0

def mouse_callback(event, x, y, flags, param):
    """
    Permite arrastrar un rectángulo en la ventana "Original".
    Actualiza variables globales del ROI.
    """
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

def draw_hud(vis, lower_skin, upper_skin, current_label):
    """
    Dibuja las instrucciones de teclado y la etiqueta activa para guardar.
    """
    hud = (
        "c=calibrar | r=reset | g=guardar | a=append seq | "
        "p=print+save seq | q=salir | [0-5]=cambiar label"
    )
    cv2.putText(vis, hud, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(vis,
                f"Label actual: {current_label}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200,255,200), 2, cv2.LINE_AA)

    if lower_skin is not None:
        txt = (
            f"HSV low:{tuple(int(v) for v in lower_skin)} "
            f"up:{tuple(int(v) for v in upper_skin)}"
        )
        cv2.putText(vis,
                    txt,
                    (10, PREVIEW_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180,255,180), 1, cv2.LINE_AA)

def draw_roi_rectangle(vis):
    """
    Enseña visualmente el ROI que usas para calibrar piel (el que marcas con el ratón).
    """
    global roi_selecting, roi_defined, x_start, y_start, x_end, y_end
    if roi_selecting or roi_defined:
        cv2.rectangle(vis,
                      (x_start, y_start),
                      (x_end,   y_end),
                      (0, 255, 255), 2)

def draw_hand_box(vis, mask):
    """
    Dibuja en 'vis' el rectángulo de mínima área (minAreaRect)
    que envuelve el contorno más grande de la máscara binaria 'mask'.

    Esto es puramente visual: sirve para enseñar qué región de la mano
    estamos usando para normalizar/recortar.
    """
    # Encontrar contorno más grande
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:
        return

    # minAreaRect -> caja rotada
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)        # puntos float32 (4x2)
    box = box.astype(np.int32)       # convertir a enteros

    # Pintar la caja en la imagen visual (lineas verdes)
    cv2.polylines(vis, [box], isClosed=True, color=(0,255,0), thickness=2)

def draw_prediction(vis, label_to_show, best_dist):
    """
    Pinta la predicción actual suavizada (o '????' si es baja confianza).
    """
    if label_to_show is None:
        return
    txt = f"Gesto: {label_to_show} (dist={best_dist:.2f})"
    cv2.putText(vis,
                txt,
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis,
                txt,
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 1, cv2.LINE_AA)

def append_action(acciones, label_to_add):
    """
    Añade el gesto actual a la lista de acciones (como número),
    siempre y cuando no sea '????' ni None.
    """
    if label_to_add is None:
        print("[WARN] No hay gesto detectado para añadir.")
        return acciones

    if label_to_add == "????":
        print("[WARN] Gesto con baja confianza / inestable, no añado a la secuencia.")
        return acciones

    acciones.append(LABEL_TO_NUM.get(label_to_add, label_to_add))
    print(f"[INFO] Añadido gesto '{label_to_add}' -> {acciones[-1]}")
    print(f"[INFO] Acciones actuales: {acciones}")
    return acciones
