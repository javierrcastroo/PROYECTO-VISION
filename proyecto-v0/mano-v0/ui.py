# ui.py
import cv2

# estado global del ROI
roi_seleccionando = False
roi_definido = False
x_inicio = y_inicio = x_fin = y_fin = 0


def callback_raton(evento, x, y, flags, param):
    global roi_seleccionando, roi_definido, x_inicio, y_inicio, x_fin, y_fin
    if evento == cv2.EVENT_LBUTTONDOWN:
        roi_seleccionando = True
        roi_definido = False
        x_inicio, y_inicio = x, y
        x_fin, y_fin = x, y
    elif evento == cv2.EVENT_MOUSEMOVE and roi_seleccionando:
        x_fin, y_fin = x, y
    elif evento == cv2.EVENT_LBUTTONUP:
        roi_seleccionando = False
        roi_definido = True
        x_fin, y_fin = x, y


def dibujar_rectangulo_roi(imagen):
    if roi_seleccionando or roi_definido:
        cv2.rectangle(
            imagen,
            (x_inicio, y_inicio),
            (x_fin, y_fin),
            (0, 255, 255),
            2,
        )


def dibujar_hud(imagen, piel_inferior, piel_superior, etiqueta_actual):
    hud = "ROI: arrastra | 'c' calib mano | 'g' guarda muestra | 'q' salir"
    cv2.putText(
        imagen,
        hud,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        imagen,
        f"Label actual: {etiqueta_actual}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 255, 180),
        1,
        cv2.LINE_AA,
    )

    if piel_inferior is not None and piel_superior is not None:
        texto_base = f"HSV low:{tuple(int(v) for v in piel_inferior)} up:{tuple(int(v) for v in piel_superior)}"
        cv2.putText(
            imagen,
            texto_base,
            (10, imagen.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 200, 255),
            1,
            cv2.LINE_AA,
        )


def dibujar_prediccion(imagen, etiqueta, distancia):
    if etiqueta is None:
        return
    texto = f"Gesto: {etiqueta} (dist={distancia:.2f})"
    cv2.putText(
        imagen,
        texto,
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        imagen,
        texto,
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def dibujar_caja_mano(imagen, mascara):
    # dibuja rect mínimo alrededor de la mano (para debug)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return
    contorno = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(contorno) < 100:
        return
    rect = cv2.minAreaRect(contorno)
    caja = cv2.boxPoints(rect)
    caja = caja.astype(int)
    cv2.polylines(imagen, [caja], True, (255, 0, 255), 2)


def dibujar_estado_secuencia(imagen, acciones, estado_captura, pendiente, lineas_estado, progreso):
    y = 100

    def _formatear_secuencia(secuencia):
        if not secuencia:
            return "(vacia)"
        mapa_letras = {
            "0dedos": "A",
            "1dedo": "B",
            "2dedos": "C",
            "3dedos": "D",
            "4dedos": "E",
        }
        display = list(secuencia)
        if display:
            first = display[0]
            if first in mapa_letras:
                display[0] = mapa_letras[first]
        return ", ".join(display)

    texto_secuencia = _formatear_secuencia(acciones)
    cv2.putText(
        imagen,
        f"Secuencia (max 2): {texto_secuencia}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )
    y += 20
    cv2.putText(
        imagen,
        f"Estado: {estado_captura}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 255),
        1,
        cv2.LINE_AA,
    )
    if pendiente:
        y += 20
        cv2.putText(
            imagen,
            f"Pendiente: {pendiente}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 255, 200),
            1,
            cv2.LINE_AA,
        )
    for linea in lineas_estado[:2]:
        if not linea:
            continue
        y += 20
        cv2.putText(
            imagen,
            linea,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    bar_x = 10
    bar_y = y + 30
    bar_w = 200
    bar_h = 10
    cv2.rectangle(
        imagen,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h),
        (100, 100, 100),
        1,
    )
    fill_w = int(bar_w * max(0.0, min(1.0, progreso)))
    cv2.rectangle(
        imagen,
        (bar_x, bar_y),
        (bar_x + fill_w, bar_y + bar_h),
        (0, 200, 0),
        -1,
    )
    cv2.putText(
        imagen,
        f"Ventana 150f: {int(progreso * 100)}%",
        (bar_x + 5, bar_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 200, 0),
        1,
        cv2.LINE_AA,
    )
