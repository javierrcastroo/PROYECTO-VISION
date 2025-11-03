import os
import cv2
import numpy as np

# Directorio donde guardamos las muestras etiquetadas y las secuencias JSON
SAVE_DIR = "gestures"
os.makedirs(SAVE_DIR, exist_ok=True)

# Si True, intentamos reconocer en vivo usando la galería
RECOGNIZE_MODE = True

# Tamaño del preview de cámara
PREVIEW_W, PREVIEW_H = 640, 480

# Tamaño normalizado de la mano (warp -> resize)
NORMALIZED_SIZE = 200

# Kernels morfológicos para limpiar la máscara
KERNEL_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

# Mapeo gesto → número, para la lista de acciones capturadas con 'a'
LABEL_TO_NUM = {
    "0dedos": 0,
    "1dedo": 1,
    "2dedos": 2,
    "3dedos": 3,
    "4dedos": 4,
    "5dedos": 5,
    "ok": "OK",
    "cool": "Cool",
    "demond" : "Demond"

}

# Umbral de confianza:
# Si la distancia media de los k vecinos (best_dist) es MAYOR que esto,
# consideramos que la predicción no es fiable y devolvemos "????".
CONFIDENCE_THRESHOLD = 1
