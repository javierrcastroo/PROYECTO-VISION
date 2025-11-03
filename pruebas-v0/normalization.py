import numpy as np
import cv2
from config import NORMALIZED_SIZE

def extract_normalized_hand(mask, output_size=NORMALIZED_SIZE):
    """
    Dada una máscara binaria (0/255) de la mano:
      1. Encuentra el contorno más grande.
      2. Calcula el rectángulo de área mínima (minAreaRect).
      3. Warp (perspectiva) para alinear ese rectángulo.
      4. Reescala a (output_size x output_size).

    Devuelve:
      norm_mask: máscara normalizada (output_size x output_size) en 0/255
      contour_norm: contorno mayor ya en coords normalizadas
    o (None, None) si no hay mano válida o algo sale raro.
    """
    # Buscar contorno más grande
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)

    # Evitar mierdecilla mínima (ruido)
    if cv2.contourArea(c) < 100:
        return None, None

    # Rectángulo de área mínima
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)  # 4 puntos float (x,y)
    # np.int0 ya no existe en numpy moderno; usamos int32
    box = box.astype(np.int32)

    # Reordenar los 4 puntos como tl, tr, br, bl
    pts = box.astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    src = np.array([tl, tr, br, bl], dtype="float32")

    # Calcular ancho/alto estimados del rectángulo alineado
    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))

    # Si el recorte potencial es demasiado pequeño, lo descartamos
    if maxW < 5 or maxH < 5:
        return None, None

    dst = np.array([
        [0,      0],
        [maxW-1, 0],
        [maxW-1, maxH-1],
        [0,      maxH-1]
    ], dtype="float32")

    # Warpeo de la máscara original al rectángulo alineado
    try:
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(mask, M, (maxW, maxH))
    except cv2.error:
        # A veces, con geometrías degeneradas, getPerspectiveTransform / warpPerspective
        # puede fallar. No queremos que pete el loop entero por 1 frame basura.
        return None, None

    # Reescalar a tamaño fijo (NORMALIZED_SIZE x NORMALIZED_SIZE)
    norm_mask = cv2.resize(
        warped,
        (output_size, output_size),
        interpolation=cv2.INTER_NEAREST
    )

    # Volvemos a sacar el contorno más grande en la imagen normalizada
    contours_norm, _ = cv2.findContours(norm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_norm:
        return norm_mask, None

    c_norm = max(contours_norm, key=cv2.contourArea)

    return norm_mask, c_norm
