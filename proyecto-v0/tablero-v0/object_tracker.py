# object_tracker.py
import cv2
import numpy as np

# color por defecto del barco largo (lo calibras con '2')
RANGO_INFERIOR_BARCO_DOBLE_DEFECTO = np.array([0, 120, 80], dtype=np.uint8)
RANGO_SUPERIOR_BARCO_DOBLE_DEFECTO = np.array([15, 255, 255], dtype=np.uint8)

# color por defecto de los barcos de una casilla (lo calibras con '1')
RANGO_INFERIOR_BARCO_SIMPLE_DEFECTO = np.array([100, 80, 80], dtype=np.uint8)
RANGO_SUPERIOR_BARCO_SIMPLE_DEFECTO = np.array([125, 255, 255], dtype=np.uint8)

# color por defecto de la munición (lo calibras con 'm')
RANGO_INFERIOR_MUNICION_DEFECTO = np.array([40, 80, 60], dtype=np.uint8)
RANGO_SUPERIOR_MUNICION_DEFECTO = np.array([80, 255, 255], dtype=np.uint8)

# color por defecto del origen (luego lo calibras con 'r')
RANGO_INFERIOR_ORIGEN_DEFECTO = np.array([90, 120, 80], dtype=np.uint8)   # azul por defecto
RANGO_SUPERIOR_ORIGEN_DEFECTO = np.array([130, 255, 255], dtype=np.uint8)

rango_inferior_barco_doble = RANGO_INFERIOR_BARCO_DOBLE_DEFECTO.copy()
rango_superior_barco_doble = RANGO_SUPERIOR_BARCO_DOBLE_DEFECTO.copy()

rango_inferior_barco_simple = RANGO_INFERIOR_BARCO_SIMPLE_DEFECTO.copy()
rango_superior_barco_simple = RANGO_SUPERIOR_BARCO_SIMPLE_DEFECTO.copy()

rango_inferior_origen = RANGO_INFERIOR_ORIGEN_DEFECTO.copy()
rango_superior_origen = RANGO_SUPERIOR_ORIGEN_DEFECTO.copy()

rango_inferior_municion = RANGO_INFERIOR_MUNICION_DEFECTO.copy()
rango_superior_municion = RANGO_SUPERIOR_MUNICION_DEFECTO.copy()


def _calibrar_desde_roi(hsv_roi, p_bajo=5, p_alto=95, margen_h=3, margen_sv=20):
    H = hsv_roi[:,:,0].reshape(-1)
    S = hsv_roi[:,:,1].reshape(-1)
    V = hsv_roi[:,:,2].reshape(-1)

    mask_validos = (V > 40) & (S > 20)
    if mask_validos.sum() > 200:
        H, S, V = H[mask_validos], S[mask_validos], V[mask_validos]

    h_lo, h_hi = np.percentile(H, [p_bajo, p_alto]).astype(int)
    s_lo, s_hi = np.percentile(S, [p_bajo, p_alto]).astype(int)
    v_lo, v_hi = np.percentile(V, [p_bajo, p_alto]).astype(int)

    h_lo = max(0, h_lo - margen_h)
    h_hi = min(179, h_hi + margen_h)
    s_lo = max(0, s_lo - margen_sv)
    s_hi = min(255, s_hi + margen_sv)
    v_lo = max(0, v_lo - margen_sv)
    v_hi = min(255, v_hi + margen_sv)

    inferior = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    superior = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
    return inferior, superior


def calibrar_color_barco_doble_desde_roi(hsv_roi):
    return _calibrar_desde_roi(hsv_roi)


def calibrar_color_barco_simple_desde_roi(hsv_roi):
    return _calibrar_desde_roi(hsv_roi)


def calibrar_color_origen_desde_roi(hsv_roi):
    return _calibrar_desde_roi(hsv_roi)


def calibrar_color_municion_desde_roi(hsv_roi):
    return _calibrar_desde_roi(hsv_roi)


def detectar_puntos_coloreados_global(frame_hsv, rango_inferior, rango_superior, max_objetos=8, area_minima=40):
    """Detecta blobs del color dado en todo el frame HSV."""
    mask = cv2.inRange(frame_hsv, rango_inferior, rango_superior)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = conservar_componentes_mayores(mask, k=max_objetos, min_area=area_minima)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) < area_minima:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))
        if len(centers) >= max_objetos:
            break

    return centers, mask


def conservar_componentes_mayores(mask, k=4, min_area=50):
    """
    Devuelve una máscara nueva con solo las k componentes más grandes
    por área (en píxeles) que superen min_area.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    mascara_filtrada = np.zeros_like(mask)
    guardados = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        cv2.drawContours(mascara_filtrada, [c], -1, 255, -1)
        guardados += 1
        if guardados >= k:
            break
    return mascara_filtrada


def detectar_puntos_coloreados_en_tablero(frame_hsv, cuadrilatero_tablero, rango_inferior, rango_superior,
                                          max_objetos=4, area_minima=50):
    """
    hsv_frame: frame del tablero en HSV
    board_quad: 4x2 float32 (tl,tr,br,bl)
    lower/upper: rango HSV del objeto/origen

    Devuelve:
      centers: lista de (x,y) en coordenadas de imagen
      mask: máscara de ese color (para debug)
    """
    # 1) máscara por color
    mask = cv2.inRange(frame_hsv, rango_inferior, rango_superior)

    # 2) pequeña limpieza morfológica
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((3,3), np.uint8), iterations=1)

    # 3) quedarnos solo con las k componentes más grandes
    mask = conservar_componentes_mayores(mask, k=max_objetos, min_area=area_minima)

    # 4) contornos finales
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], mask

    poligono_tablero = np.array(cuadrilatero_tablero, dtype=np.float32)

    # ordenar por área desc
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    centers = []
    for c in contours:
        if cv2.contourArea(c) < area_minima:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # comprobar si está dentro del tablero
        if cv2.pointPolygonTest(poligono_tablero, (cx, cy), False) >= 0:
            centers.append((cx, cy))

        if len(centers) >= max_objetos:
            break

    return centers, mask
