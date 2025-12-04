# board_tracker.py
import cv2
import numpy as np

# nº de casillas del tablero
CASILLAS_TABLERO = 5
# tamaño real de una casilla (cm) – pon el tuyo
TAMANO_CASILLA_CM = 3.7

# rango por defecto
RANGO_INFERIOR_DEFECTO = np.array([5, 80, 80], dtype=np.uint8)
RANGO_SUPERIOR_DEFECTO = np.array([25, 255, 255], dtype=np.uint8)

# estos son los que vamos cambiando con 'b'
rango_inferior_actual = RANGO_INFERIOR_DEFECTO.copy()
rango_superior_actual = RANGO_SUPERIOR_DEFECTO.copy()


def calibrar_color_tablero_desde_roi(hsv_roi, p_bajo=5, p_alto=95):
    H = hsv_roi[:, :, 0].reshape(-1)
    S = hsv_roi[:, :, 1].reshape(-1)
    V = hsv_roi[:, :, 2].reshape(-1)

    h_lo, h_hi = np.percentile(H, [p_bajo, p_alto]).astype(int)
    s_lo, s_hi = np.percentile(S, [p_bajo, p_alto]).astype(int)
    v_lo, v_hi = np.percentile(V, [p_bajo, p_alto]).astype(int)

    h_lo = max(0, h_lo - 3)
    h_hi = min(179, h_hi + 3)
    s_lo = max(0, s_lo - 20)
    s_hi = min(255, s_hi + 20)
    v_lo = max(0, v_lo - 20)
    v_hi = min(255, v_hi + 20)

    inferior = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    superior = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
    return inferior, superior


def ordenar_puntos(puntos):
    suma = puntos.sum(axis=1)
    diferencia = np.diff(puntos, axis=1)
    tl = puntos[np.argmin(suma)]
    br = puntos[np.argmax(suma)]
    tr = puntos[np.argmin(diferencia)]
    bl = puntos[np.argmax(diferencia)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def dibujar_cuadricula_en_cuadrilatero(imagen, cuadrilatero, casillas=CASILLAS_TABLERO):
    tl, tr, br, bl = cuadrilatero
    # horizontales
    for i in range(casillas + 1):
        t = i / casillas
        p1 = tl * (1 - t) + bl * t
        p2 = tr * (1 - t) + br * t
        cv2.line(imagen, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 0, 0), 1)
    # verticales
    for j in range(casillas + 1):
        t = j / casillas
        p1 = tl * (1 - t) + tr * t
        p2 = bl * (1 - t) + br * t
        cv2.line(imagen, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 255, 0), 1)


def detectar_tablero(frame, matriz_camara=None, coeficientes_distorsion=None):
    """
    Versión original: devuelve un solo tablero.
    La dejamos por compatibilidad.
    """
    global rango_inferior_actual, rango_superior_actual

    if matriz_camara is not None and coeficientes_distorsion is not None:
        frame = cv2.undistort(frame, matriz_camara, coeficientes_distorsion)

    vis = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, rango_inferior_actual, rango_superior_actual)
    mask_show = mask.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask_unida = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_unida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cv2.putText(vis, "Buscando tablero...", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return vis, False, None, None, mask_show, None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 5000:
        return vis, False, None, None, mask_show, None

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    quad = ordenar_puntos(box.astype(np.float32))

    dibujar_cuadricula_en_cuadrilatero(vis, quad, CASILLAS_TABLERO)

    tl, tr, br, bl = quad
    top_mid = (tl + tr) / 2.0
    bot_mid = (bl + br) / 2.0
    height_px = float(np.linalg.norm(top_mid - bot_mid))

    ratio_cm_per_pix = None
    if height_px > 1e-3:
        pix_per_square = height_px / CASILLAS_TABLERO
        ratio_cm_per_pix = TAMANO_CASILLA_CM / pix_per_square

    if ratio_cm_per_pix is not None:
        est_height_cm = height_px * ratio_cm_per_pix
        cv2.putText(vis, f"{ratio_cm_per_pix:.4f} cm/pix | alto={est_height_cm:.1f}cm",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    return vis, True, ratio_cm_per_pix, height_px, mask_show, quad


def detectar_tableros_multiples(frame, matriz_camara=None, coeficientes_distorsion=None, max_tableros=2):
    """
    NUEVO: detecta hasta `max_boards` tableros del mismo color.
    Devuelve:
      vis        -> imagen con todos dibujados
      boards     -> lista de dicts { 'quad':..., 'ratio':..., 'height_px':... }
      mask_show  -> máscara de color original
    """
    global rango_inferior_actual, rango_superior_actual

    if matriz_camara is not None and coeficientes_distorsion is not None:
        frame = cv2.undistort(frame, matriz_camara, coeficientes_distorsion)

    vis = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, rango_inferior_actual, rango_superior_actual)
    mask_show = mask.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask_unida = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_unida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cv2.putText(vis, "Buscando tableros...", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return vis, [], mask_show

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    tableros = []

    for idx, c in enumerate(contours[:max_tableros]):
        if cv2.contourArea(c) < 5000:
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        quad = ordenar_puntos(box.astype(np.float32))

        dibujar_cuadricula_en_cuadrilatero(vis, quad, CASILLAS_TABLERO)

        tl, tr, br, bl = quad
        top_mid = (tl + tr) / 2.0
        bot_mid = (bl + br) / 2.0
        height_px = float(np.linalg.norm(top_mid - bot_mid))

        ratio = None
        if height_px > 1e-3:
            pix_per_square = height_px / CASILLAS_TABLERO
            ratio = TAMANO_CASILLA_CM / pix_per_square

        cv2.putText(vis, f"T{idx+1}",
                    tuple(quad[0].astype(int) + np.array([5, 15])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        tableros.append({
            "quad": quad,
            "ratio": ratio,
            "height_px": height_px,
        })

    return vis, tableros, mask_show
