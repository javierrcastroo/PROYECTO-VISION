import numpy as np
import cv2
from normalization import extract_normalized_hand

def compute_hu_moments(contour):
    """
    Devuelve los 7 Hu moments del contorno.
    """
    M = cv2.moments(contour)
    hu = cv2.HuMoments(M).flatten()
    # Versión log-escala opcional:
    # hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-9)
    return hu

def compute_radial_signature(mask, contour, num_angles=36):
    """
    Firma radial normalizada:
      - calcula el centroide del contorno
      - dispara rayos en 36 ángulos
      - mide distancia hasta salir de la máscara
      - normaliza por la distancia máxima
    """
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

    max_distances = np.array(max_distances, dtype=np.float32)
    m = max_distances.max() if max_distances.max() > 0 else 1.0
    radial_norm = max_distances / m  # valores 0-1 relativos

    return radial_norm

def compute_feature_vector(mask):
    """
    1. Normaliza la mano espacialmente (warp+resize).
    2. Calcula:
       - Hu moments (7,)
       - Firma radial (36,)
    3. Devuelve feat (~43 dims) o None si no hay mano válida.
    """
    norm_mask, contour_norm = extract_normalized_hand(mask)
    if norm_mask is None or contour_norm is None:
        return None

    hu = compute_hu_moments(contour_norm)                           # (7,)
    radial = compute_radial_signature(norm_mask, contour_norm, 36)  # (36,)

    feat = np.concatenate([hu, radial.astype(np.float64)], axis=0)
    return feat
