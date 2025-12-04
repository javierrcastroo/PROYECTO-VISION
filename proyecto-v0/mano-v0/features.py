# features.py
import cv2
import numpy as np
from normalization import extraer_mano_normalizada


def calcular_momentos_hu(contorno):
    momentos = cv2.moments(contorno)
    momentos_hu = cv2.HuMoments(momentos).flatten()
    return momentos_hu


def calcular_firma_radial(mascara, contorno, num_angulos=36):
    momentos = cv2.moments(contorno)
    if momentos["m00"] == 0:
        return np.zeros(num_angulos, dtype=np.float32)

    centro_x = momentos["m10"] / momentos["m00"]
    centro_y = momentos["m01"] / momentos["m00"]

    alto, ancho = mascara.shape[:2]
    angulos = np.linspace(0, 2 * np.pi, num_angulos, endpoint=False)
    distancias = []
    for theta in angulos:
        delta_x = np.cos(theta)
        delta_y = np.sin(theta)
        distancia = 0.0
        while True:
            x = int(round(centro_x + distancia * delta_x))
            y = int(round(centro_y + distancia * delta_y))
            if x < 0 or x >= ancho or y < 0 or y >= alto:
                break
            if mascara[y, x] == 0:
                break
            distancia += 1.0
        distancias.append(distancia)

    distancias = np.array(distancias, dtype=np.float32)
    maximo = distancias.max() if distancias.max() > 0 else 1.0
    return distancias / maximo


def calcular_vector_caracteristicas(mascara):
    mascara_norm, contorno = extraer_mano_normalizada(mascara)
    if mascara_norm is None or contorno is None:
        return None

    # encontrar contorno en la máscara normalizada
    contornos, _ = cv2.findContours(mascara_norm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    contorno_principal = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(contorno_principal) < 50:
        return None

    momentos_hu = calcular_momentos_hu(contorno_principal)
    firma_radial = calcular_firma_radial(mascara_norm, contorno_principal, 36)

    vector = np.concatenate([momentos_hu.astype(np.float64), firma_radial.astype(np.float64)], axis=0)
    return vector  # ~43 dims
