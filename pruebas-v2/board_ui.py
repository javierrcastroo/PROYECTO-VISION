# board_ui.py
import cv2

# ====== ROI PARA COLOR DEL TABLERO (clic izquierdo) ======
board_roi_selecting = False
board_roi_defined = False
bx_start = by_start = bx_end = by_end = 0

# ====== PUNTOS DE MEDIDA (clic derecho) ======
measure_points = []   # lista de (x, y)

def board_mouse_callback(event, x, y, flags, param):
    """
    - Botón izquierdo: definir ROI del tablero (para calibrar HSV con 'b')
    - Botón derecho: añadir punto de medida
    """
    global board_roi_selecting, board_roi_defined
    global bx_start, by_start, bx_end, by_end
    global measure_points

    if event == cv2.EVENT_LBUTTONDOWN:
        board_roi_selecting = True
        board_roi_defined = False
        bx_start, by_start = x, y
        bx_end, by_end = x, y

    elif event == cv2.EVENT_MOUSEMOVE and board_roi_selecting:
        bx_end, by_end = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        board_roi_selecting = False
        board_roi_defined = True
        bx_end, by_end = x, y

    elif event == cv2.EVENT_RBUTTONDOWN:
        # añadir punto de medida
        measure_points.append((x, y))
        # si hay más de 2, reseteamos para empezar otra medida
        if len(measure_points) > 2:
            measure_points = [(x, y)]  # empezamos de nuevo solo con este
        # print(f"[INFO] punto de medida añadido: {(x, y)}")


def draw_board_roi(img):
    """
    Dibuja el rectángulo del ROI si se está seleccionando.
    """
    if board_roi_selecting or board_roi_defined:
        cv2.rectangle(
            img,
            (bx_start, by_start),
            (bx_end,   by_end),
            (0, 255, 255),
            2
        )


def draw_measure_points(img):
    """
    Dibuja los puntos de medida (clic derecho) sobre la imagen del tablero.
    """
    for i, (x, y) in enumerate(measure_points):
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, f"P{i+1}", (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
