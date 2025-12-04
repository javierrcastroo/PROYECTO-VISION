# board_ui.py
import cv2

# ====== ROI PARA COLOR DEL TABLERO (clic izquierdo) ======
seleccion_roi_tablero = False
roi_tablero_definido = False
x_inicio_roi_tablero = y_inicio_roi_tablero = x_fin_roi_tablero = y_fin_roi_tablero = 0

# ====== PUNTOS DE MEDIDA (clic derecho) ======
puntos_medicion = []   # lista de (x, y)

# ====== ESTADOS DEL TABLERO ======
_TEXTOS_ESTADO_JUEGO = {
    "standby": "Estado: standby",
    "captura": "Estado: captura",
    "turno": "Estado: turno",
}

_TEXTOS_RESULTADO_ATAQUE = {
    "invalido": "Ataque invalido",
    "agua": "Ataque en agua",
    "tocado": "Ataque tocado",
    "hundido": "Ataque hundido",
    None: "Ataque no disponible",
}

_estado_juego_actual = _TEXTOS_ESTADO_JUEGO["standby"]
_resultado_ataque_actual = _TEXTOS_RESULTADO_ATAQUE[None]


def callback_raton_tablero(evento, x, y, flags, param):
    """
    - Botón izquierdo: definir ROI del tablero (para calibrar HSV con 'b', 'o', 'm', ...)
    - Botón derecho: añadir punto de medida
    """
    global seleccion_roi_tablero, roi_tablero_definido
    global x_inicio_roi_tablero, y_inicio_roi_tablero, x_fin_roi_tablero, y_fin_roi_tablero
    global puntos_medicion

    if evento == cv2.EVENT_LBUTTONDOWN:
        seleccion_roi_tablero = True
        roi_tablero_definido = False
        x_inicio_roi_tablero, y_inicio_roi_tablero = x, y
        x_fin_roi_tablero, y_fin_roi_tablero = x, y

    elif evento == cv2.EVENT_MOUSEMOVE and seleccion_roi_tablero:
        x_fin_roi_tablero, y_fin_roi_tablero = x, y

    elif evento == cv2.EVENT_LBUTTONUP:
        seleccion_roi_tablero = False
        roi_tablero_definido = True
        x_fin_roi_tablero, y_fin_roi_tablero = x, y

    elif evento == cv2.EVENT_RBUTTONDOWN:
        # añadir punto de medida
        puntos_medicion.append((x, y))
        # si hay más de 2, reseteamos para empezar otra medida
        if len(puntos_medicion) > 2:
            puntos_medicion = [(x, y)]  # empezamos de nuevo solo con este


def dibujar_roi_tablero(img):
    """
    Dibuja el rectángulo del ROI si se está seleccionando.
    """
    if seleccion_roi_tablero or roi_tablero_definido:
        cv2.rectangle(
            img,
            (x_inicio_roi_tablero, y_inicio_roi_tablero),
            (x_fin_roi_tablero,   y_fin_roi_tablero),
            (0, 255, 255),
            2
        )


def dibujar_puntos_medicion(img):
    """
    Dibuja los puntos de medida (clic derecho) sobre la imagen del tablero.
    """
    for i, (x, y) in enumerate(puntos_medicion):
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, f"P{i+1}", (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)


def establecer_estado_juego(clave_estado):
    """Actualiza el texto mostrado para el estado del tablero."""
    global _estado_juego_actual
    _estado_juego_actual = _TEXTOS_ESTADO_JUEGO.get(
        clave_estado, _TEXTOS_ESTADO_JUEGO["standby"]
    )


def establecer_resultado_ataque(clave_resultado):
    """Actualiza el mensaje del ultimo resultado de ataque."""
    global _resultado_ataque_actual
    _resultado_ataque_actual = _TEXTOS_RESULTADO_ATAQUE.get(
        clave_resultado, _TEXTOS_RESULTADO_ATAQUE[None]
    )


def dibujar_estado_partida(img):
    """Dibuja los textos de estado y ultimo ataque en la esquina inferior derecha."""
    if img is None:
        return

    lines = [_estado_juego_actual, _resultado_ataque_actual]
    margin = 10
    x = img.shape[1] - 300
    y = img.shape[0] - 20

    for line in reversed(lines):
        cv2.putText(
            img,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y -= 20 + margin // 2


def dibujar_hud_tablero(img):
    """
    HUD con las teclas específicas del tablero.
    """
    y0 = 20
    dy = 18
    cv2.putText(img, "TABLERO:", (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img, "b: calibrar color tablero (ROI izq)", (10, y0 + dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img, "r: calibrar ORIGEN (ROI marcador)", (10, y0 + 2*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img, "2: calibrar BARCO x2 (ROI barco)", (10, y0 + 3*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img, "1: calibrar BARCO x1 (ROI barco)", (10, y0 + 4*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img, "m: calibrar MUN (ROI municion)", (10, y0 + 5*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img, "Click izq: definir ROI", (10, y0 + 6*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,255,200), 1, cv2.LINE_AA)
    cv2.putText(img, "Click dcho: punto de medida", (10, y0 + 7*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,255,200), 1, cv2.LINE_AA)


def dibujar_resultado_validacion(img, quad, text, is_valid):
    if quad is None:
        return
    color = (0, 255, 0) if is_valid else (0, 0, 255)
    tl = quad[0]
    x = int(tl[0])
    y = int(tl[1]) - 10
    cv2.putText(
        img,
        text,
        (x, max(20, y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
