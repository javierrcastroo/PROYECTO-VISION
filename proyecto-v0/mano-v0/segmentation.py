# segmentation.py
import cv2
import numpy as np

KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def limitar(valor, minimo, maximo):
    return int(max(minimo, min(maximo, valor)))


def calibrar_desde_roi(hsv_roi, p_bajo=5, p_alto=95, margen_h=5, margen_sv=20):
    tono = hsv_roi[:, :, 0].reshape(-1)
    saturacion = hsv_roi[:, :, 1].reshape(-1)
    valor = hsv_roi[:, :, 2].reshape(-1)

    mascara_valida = (valor > 30) & (valor < 245) & (saturacion > 20)
    if mascara_valida.sum() > 200:
        tono, saturacion, valor = tono[mascara_valida], saturacion[mascara_valida], valor[mascara_valida]

    h_lo, h_hi = np.percentile(tono, [p_bajo, p_alto]).astype(int)
    s_lo, s_hi = np.percentile(saturacion, [p_bajo, p_alto]).astype(int)
    v_lo, v_hi = np.percentile(valor, [p_bajo, p_alto]).astype(int)

    h_lo = limitar(h_lo - margen_h, 0, 179)
    h_hi = limitar(h_hi + margen_h, 0, 179)
    s_lo = limitar(s_lo - margen_sv, 0, 255)
    s_hi = limitar(s_hi + margen_sv, 0, 255)
    v_lo = limitar(v_lo - margen_sv, 0, 255)
    v_hi = limitar(v_hi + margen_sv, 0, 255)

    if h_lo > h_hi:
        h_lo, h_hi = 0, max(h_lo, h_hi)

    inferior = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    superior = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
    return inferior, superior


def mascara_componente_mas_grande(binaria):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    if num_labels <= 1:
        return binaria
    etiqueta_mas_grande = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == etiqueta_mas_grande, 255, 0).astype(np.uint8)


def segmentar_mascara_mano(frame_hsv, piel_inferior, piel_superior):
    if piel_inferior is None or piel_superior is None:
        return np.zeros(frame_hsv.shape[:2], dtype=np.uint8)

    mascara_bruta = cv2.inRange(frame_hsv, piel_inferior, piel_superior)
    mascara = cv2.GaussianBlur(mascara_bruta, (5, 5), 0)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, KERNEL_OPEN)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, KERNEL_CLOSE)
    mascara_principal = mascara_componente_mas_grande(mascara)
    return mascara_principal
