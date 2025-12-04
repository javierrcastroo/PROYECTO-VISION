# aruco_utils.py
import cv2
import numpy as np
import board_state

# Diccionario de ArUco que vamos a usar
_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
_ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# ID del marcador que usaremos como origen global
ARUCO_ORIGEN_ID = 2


def detectar_origen_aruco(frame, aruco_id=ARUCO_ORIGEN_ID, dibujar=True):
    """
    Detecta en 'frame' un marcador ArUco con ID = aruco_id.
    Si lo encuentra:
      - devuelve True, (cx, cy)
      - opcionalmente lo dibuja
    Si no:
      - devuelve False, None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector = cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return False, None

    ids = ids.flatten()
    for corner, marker_id in zip(corners, ids):
        if marker_id == aruco_id:
            pts = corner.reshape((4, 2))
            (tl, tr, br, bl) = pts
            # centro del marcador
            cx = int((tl[0] + br[0]) / 2.0)
            cy = int((tl[1] + br[1]) / 2.0)

            if dibujar:
                cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(frame, f"ArUco {marker_id}", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            return True, (cx, cy)

    return False, None


def actualizar_origen_global_desde_aruco(frame, aruco_id=ARUCO_ORIGEN_ID):
    """
    Llama a detectar_origen_aruco y, si lo encuentra, actualiza board_state.ORIGEN_GLOBAL.
    Si no lo ve un frame, aguanta unos cuantos antes de borrarlo.
    """
    ok, pt = detectar_origen_aruco(frame, aruco_id=aruco_id, dibujar=True)
    if ok:
        board_state.ORIGEN_GLOBAL = pt
        board_state.ORIGEN_GLOBAL_FALLOS = 0
    else:
        board_state.ORIGEN_GLOBAL_FALLOS += 1
        if board_state.ORIGEN_GLOBAL_FALLOS > board_state.ORIGEN_GLOBAL_MAX_FALLOS:
            board_state.ORIGEN_GLOBAL = None
