# normalization.py
import cv2
import numpy as np

def extract_normalized_hand(mask, size=200):
    """
    Recibe una máscara 0/255 y devuelve:
      - mask_rot_resized: máscara rotada y reescalada a (size, size)
      - contour_rot: contorno principal ya en ese espacio
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:
        return None, None

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = box.astype(int)

    angle = rect[2]
    if angle < -45:
        angle += 90

    (h, w) = mask.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    x, y, rw, rh = cv2.boundingRect(cv2.transform(np.array([c]), M)[0])
    hand_roi = rotated[y:y+rh, x:x+rw]
    if hand_roi.size == 0:
        return None, None

    resized = cv2.resize(hand_roi, (size, size), interpolation=cv2.INTER_NEAREST)

    return resized, c  # devolvemos también el contorno original por si se usa

