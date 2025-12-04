# board_main.py
import cv2
import json
import os
import glob
import time
import numpy as np
from collections import Counter

CAPTURE_BUFFER_FRAMES = 150

from board_config import (
    TAMANO_WARP,
    RUTA_PARAMETROS_CAMARA_TABLERO,
    USAR_CORRECCION_DISTORSION_TABLERO,
)
import board_ui
import board_state
import board_processing as bp
import aruco_utils
import battleship_logic

ATTACKS_DIR = os.path.join(os.path.dirname(__file__), "..", "ataques")
LAST_RESULT_FILE = os.path.join(ATTACKS_DIR, "last_result.json")
RESTART_FILE = os.path.join(ATTACKS_DIR, "restart.json")
CAPTURE_FRAMES = 150

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara 1 (tablero)")

    # cargar calibracion de camara
    mtx = dist = None
    if USAR_CORRECCION_DISTORSION_TABLERO and os.path.exists(RUTA_PARAMETROS_CAMARA_TABLERO):
        data = np.load(RUTA_PARAMETROS_CAMARA_TABLERO)
        mtx = data["camera_matrix"]
        dist = data["dist_coeffs"]
        

    # dos tableros
    boards_state_list = _init_boards()

    cv2.namedWindow("Tablero")
    cv2.setMouseCallback("Tablero", board_ui.callback_raton_tablero)

    os.makedirs(ATTACKS_DIR, exist_ok=True)

    estado = "ESPERA"  # ESPERA -> CAPTURA -> JUEGO
    frames_captura_restantes = 0
    acumulacion = {"T1": [], "T2": []}
    distribuciones_estables = None
    estado_partida = None
    ataques_procesados = set()
    ultimas_lineas_estado = None
    lineas_estado = [
        "Standby: coloca barcos y calibra HSV",
        f"Pulsa 's' para fijar el layout ({CAPTURE_FRAMES} frames)",
    ]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # detectar el ORIGEN con ArUco cada frame
        aruco_utils.actualizar_origen_global_desde_aruco(frame, aruco_id=2)

        # procesar todos los tableros con el origen global actual
        vis, mask_b, mask_ship2, mask_ship1, mask_m, layouts = bp.procesar_todos_los_tableros(
            frame,
            boards_state_list,
            matriz_camara=mtx,
            coef_distorsion=dist,
            max_tableros=2,
            tamano_warp=TAMANO_WARP,
            imprimir_detecciones=False,
        )

        validation_map = {}
        for layout in layouts:
            ok, msg = battleship_logic.evaluar_tablero(layout)
            validation_map[layout["name"]] = (ok, msg)

        if estado == "CAPTURA":
            _acumular_layouts(layouts, acumulacion)
            frames_captura_restantes -= 1
            lineas_estado = [
                f"Capturando layout estable ({frames_captura_restantes} frames restantes)",
            ]
            if frames_captura_restantes <= 0:
                distribuciones_estables = _seleccionar_layouts_estables(acumulacion, boards_state_list)
                estado_partida = battleship_logic.inicializar_estado_partida(distribuciones_estables)
                estado = "JUEGO"
                lineas_estado = [
                    "Layouts fijados. Empieza la partida.",
                    "Turno inicial: T1 ataca T2",
                ]
                _registrar_layouts_estables(distribuciones_estables, boards_state_list)
                _escribir_ultimo_resultado(_capturar_estado_turno(estado_partida), lineas_estado, estado_partida)

        for slot in boards_state_list:
            if slot["name"] in validation_map and slot["last_quad"] is not None:
                ok, msg = validation_map[slot["name"]]
                board_ui.dibujar_resultado_validacion(vis, slot["last_quad"], msg, ok)

        # dibujar el origen global si lo tenemos
        if board_state.ORIGEN_GLOBAL is not None:
            gx, gy = board_state.ORIGEN_GLOBAL
            cv2.circle(vis, (int(gx), int(gy)), 10, (0, 255, 0), -1)
            cv2.putText(
                vis,
                "ORIGEN (ArUco)",
                (int(gx) + 10, int(gy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if estado == "JUEGO" and estado_partida is not None:
            if not estado_partida.get("finished"):
                lineas_estado = [
                    f"Turno de ataque: {estado_partida['current_attacker']} -> {estado_partida['current_defender']}",
                ]
            else:
                lineas_estado = [
                    f"Partida terminada. Ganador: {estado_partida.get('winner')}",
                ]

            mensaje_pendiente = _procesar_nuevos_ataques(estado_partida, ataques_procesados)
            if mensaje_pendiente:
                lineas_estado = mensaje_pendiente

            if estado_partida.get("finished") and _consumir_solicitud_reinicio():
                (
                    boards_state_list,
                    acumulacion,
                    distribuciones_estables,
                    estado_partida,
                    ataques_procesados,
                    estado,
                    lineas_estado,
                    ultimas_lineas_estado,
                    frames_captura_restantes,
                ) = _reiniciar_partida()

        board_ui.dibujar_hud_tablero(vis)
        _dibujar_lineas_estado(vis, lineas_estado)

        if lineas_estado != ultimas_lineas_estado:
            for line in lineas_estado:
                print(line)
            ultimas_lineas_estado = list(lineas_estado)

        # mostrar
        cv2.imshow("Tablero", vis)
        cv2.imshow("Mascara tablero", mask_b)
        if mask_ship2 is not None:
            cv2.imshow("Mascara barco x2", mask_ship2)
        if mask_ship1 is not None:
            cv2.imshow("Mascara barco x1", mask_ship1)
        if mask_m is not None:
            cv2.imshow("Mascara municion", mask_m)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        if key == ord("s") and estado == "ESPERA":
            frames_captura_restantes = CAPTURE_FRAMES
            acumulacion = {"T1": [], "T2": []}
            estado = "CAPTURA"
            lineas_estado = [
                f"Inicio de captura de layout durante {CAPTURE_FRAMES} frames",
            ]
        elif key == ord("r"):
            (
                boards_state_list,
                acumulacion,
                distribuciones_estables,
                estado_partida,
                ataques_procesados,
                estado,
                lineas_estado,
                ultimas_lineas_estado,
                frames_captura_restantes,
            ) = _reiniciar_partida()
        else:
            handle_keys(key, frame)

    cap.release()
    cv2.destroyAllWindows()


def handle_keys(key, frame):
   
    # b = color tablero
    # 2 = barco de dos casillas
    # 1 = barco de una casilla
    # m = color de la munición
    import board_tracker
    import object_tracker
    import board_ui as bu

    if key == ord("b"):
        if bu.roi_tablero_definido:
            x0, x1 = sorted([bu.x_inicio_roi_tablero, bu.x_fin_roi_tablero])
            y0, y1 = sorted([bu.y_inicio_roi_tablero, bu.y_fin_roi_tablero])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = board_tracker.calibrar_color_tablero_desde_roi(roi_hsv)
            board_tracker.rango_inferior_actual, board_tracker.rango_superior_actual = lo, up
            print("[INFO] calibrado TABLERO:", lo, up)
        else:
            print("[WARN] dibuja ROI en 'Tablero' primero")

    elif key == ord("2"):
        if bu.roi_tablero_definido:
            x0, x1 = sorted([bu.x_inicio_roi_tablero, bu.x_fin_roi_tablero])
            y0, y1 = sorted([bu.y_inicio_roi_tablero, bu.y_fin_roi_tablero])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = object_tracker.calibrar_color_barco_doble_desde_roi(roi_hsv)
            object_tracker.rango_inferior_barco_doble, object_tracker.rango_superior_barco_doble = lo, up
            print("[INFO] calibrado BARCO x2:", lo, up)
        else:
            print("[WARN] dibuja ROI sobre el barco largo")

    elif key == ord("1"):
        if bu.roi_tablero_definido:
            x0, x1 = sorted([bu.x_inicio_roi_tablero, bu.x_fin_roi_tablero])
            y0, y1 = sorted([bu.y_inicio_roi_tablero, bu.y_fin_roi_tablero])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = object_tracker.calibrar_color_barco_simple_desde_roi(roi_hsv)
            object_tracker.rango_inferior_barco_simple, object_tracker.rango_superior_barco_simple = lo, up
            print("[INFO] calibrado BARCO x1:", lo, up)
        else:
            print("[WARN] dibuja ROI sobre el barco corto")

    elif key == ord("m"):
        if bu.roi_tablero_definido:
            x0, x1 = sorted([bu.x_inicio_roi_tablero, bu.x_fin_roi_tablero])
            y0, y1 = sorted([bu.y_inicio_roi_tablero, bu.y_fin_roi_tablero])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = object_tracker.calibrar_color_municion_desde_roi(roi_hsv)
            object_tracker.rango_inferior_municion, object_tracker.rango_superior_municion = lo, up
            print("[INFO] calibrada MUNICION:", lo, up)
        else:
            print("[WARN] dibuja ROI sobre la municion")



def _init_boards():
    return [
        board_state.inicializar_estado_tablero("T1"),
        board_state.inicializar_estado_tablero("T2"),
    ]


def _crear_captura_layout(layout):
    return {
        "ship_two_cells": tuple(sorted(layout.get("ship_two_cells", []))),
        "ship_one_cells": tuple(sorted(layout.get("ship_one_cells", []))),
        "board_size": layout.get("board_size", 5),
    }


def _acumular_layouts(layouts, acumulacion):
    for layout in layouts:
        name = layout.get("name")
        if name not in acumulacion:
            acumulacion[name] = []
        acumulacion[name].append(_crear_captura_layout(layout))


def _seleccionar_layouts_estables(samples_map, boards_state_list):
    stable = {}
    for slot in boards_state_list:
        name = slot["name"]
        samples = samples_map.get(name, [])
        if not samples:
            stable[name] = {
                "ship_two_cells": [],
                "ship_one_cells": [],
                "board_size": 5,
            }
            continue

        # elegir el layout mas repetido
        counts = {}
        for snap in samples:
            key = (snap["ship_two_cells"], snap["ship_one_cells"])
            counts[key] = counts.get(key, 0) + 1

        best_key = max(counts.items(), key=lambda kv: kv[1])[0]
        stable[name] = {
            "ship_two_cells": list(best_key[0]),
            "ship_one_cells": list(best_key[1]),
            "board_size": samples[0].get("board_size", 5),
        }
    return stable


def _registrar_layouts_estables(stable_layouts, boards_state_list):
    if not stable_layouts:
        return

    print("[INFO] Layouts capturados tras estabilizar 150 frames:")
    for slot in boards_state_list:
        name = slot.get("name", "?")
        layout = stable_layouts.get(name, {})
        board_size = layout.get("board_size", 5)
        ship_two_cells = layout.get("ship_two_cells", [])
        ship_one_cells = layout.get("ship_one_cells", [])

        def _fmt_cells(cells):
            return ", ".join(bp._formatear_etiqueta_celda(r, c) for r, c in cells) or "(ninguno)"

        print(f"[{name}] Tablero {board_size}x{board_size}")
        print(f"    Barcos x2: {_fmt_cells(ship_two_cells)}")
        print(f"    Barcos x1: {_fmt_cells(ship_one_cells)}")

        def _log_points(tag, points):
            if not points:
                print(f"    {tag}: sin coordenadas detectadas")
                return
            for idx, (px, py) in enumerate(points, 1):
                print(f"    {tag}-{idx}: ({px:.1f}, {py:.1f})")

        _log_points("Coordenadas x2", slot.get("ship_two_points", []))
        _log_points("Coordenadas x1", slot.get("ship_one_points", []))


def _dibujar_lineas_estado(vis, lines):
    if not lines:
        return
    y = 30
    for line in lines:
        cv2.putText(
            vis,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        y += 20


def _procesar_nuevos_ataques(estado_partida, ataques_procesados):
    msgs = None
    files = sorted(glob.glob(os.path.join(ATTACKS_DIR, "T*_*.json")))
    for fp in files:
        fname = os.path.basename(fp)
        if fname in ataques_procesados:
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] no se pudo leer {fname}: {exc}")
            ataques_procesados.add(fname)
            continue

        target = payload.get("target")
        row = payload.get("row")
        col = payload.get("col")
        if target is None or row is None or col is None:
            print(f"[WARN] ataque {fname} incompleto")
            ataques_procesados.add(fname)
            continue

        result = battleship_logic.aplicar_ataque(estado_partida, target, row, col)
        ataques_procesados.add(fname)

        msgs = _formatear_resultado_ataque(result)
        _escribir_ultimo_resultado(result, msgs, estado_partida)

        try:
            os.remove(fp)
        except OSError:
            pass
    return msgs


def _consumir_solicitud_reinicio():
    if not os.path.exists(RESTART_FILE):
        return False
    try:
        os.remove(RESTART_FILE)
    except OSError:
        pass
    return True


def _limpiar_archivos_ataque_pendientes():
    try:
        for fp in glob.glob(os.path.join(ATTACKS_DIR, "T*_*.json")):
            os.remove(fp)
    except OSError:
        pass


def _reiniciar_partida():
    _reiniciar_calibracion()

    boards_state_list = _init_boards()
    acumulacion = {"T1": [], "T2": []}
    distribuciones_estables = None
    estado_partida = None
    ataques_procesados = set()
    board_state.ORIGEN_GLOBAL = None
    board_state.ORIGEN_GLOBAL_FALLOS = 0
    estado = "ESPERA"
    lineas_estado = [
        "Reinicio solicitado. Coloca de nuevo los tableros y calibra si es necesario.",
        f"Pulsa 's' para fijar el layout ({CAPTURE_FRAMES} frames)",
    ]
    frames_captura_restantes = 0

    _limpiar_archivos_ataque_pendientes()
    _escribir_ultimo_resultado(
        {"timestamp": int(time.time()), "status": "reset"},
        lineas_estado,
        estado_partida,
    )
    ultimas_lineas_estado = None

    return (
        boards_state_list,
        acumulacion,
        distribuciones_estables,
        estado_partida,
        ataques_procesados,
        estado,
        lineas_estado,
        ultimas_lineas_estado,
        frames_captura_restantes,
    )


def _reiniciar_calibracion():
    import board_tracker
    import object_tracker

    board_tracker.rango_inferior_actual = board_tracker.RANGO_INFERIOR_DEFECTO.copy()
    board_tracker.rango_superior_actual = board_tracker.RANGO_SUPERIOR_DEFECTO.copy()

    object_tracker.rango_inferior_barco_doble = object_tracker.RANGO_INFERIOR_BARCO_DOBLE_DEFECTO.copy()
    object_tracker.rango_superior_barco_doble = object_tracker.RANGO_SUPERIOR_BARCO_DOBLE_DEFECTO.copy()
    object_tracker.rango_inferior_barco_simple = object_tracker.RANGO_INFERIOR_BARCO_SIMPLE_DEFECTO.copy()
    object_tracker.rango_superior_barco_simple = object_tracker.RANGO_SUPERIOR_BARCO_SIMPLE_DEFECTO.copy()
    object_tracker.rango_inferior_municion = object_tracker.RANGO_INFERIOR_MUNICION_DEFECTO.copy()
    object_tracker.rango_superior_municion = object_tracker.RANGO_SUPERIOR_MUNICION_DEFECTO.copy()
    object_tracker.rango_inferior_origen = object_tracker.RANGO_INFERIOR_ORIGEN_DEFECTO.copy()
    object_tracker.rango_superior_origen = object_tracker.RANGO_SUPERIOR_ORIGEN_DEFECTO.copy()

    board_ui.seleccion_roi_tablero = False
    board_ui.roi_tablero_definido = False
    board_ui.x_inicio_roi_tablero = board_ui.y_inicio_roi_tablero = board_ui.x_fin_roi_tablero = board_ui.y_fin_roi_tablero = 0
    board_ui.puntos_medicion = []

    bp.limpiar_cache_visualizacion()


def _formatear_resultado_ataque(result):
    if result is None:
        return None

    estado_resultado = result.get("status")
    atacante = result.get("attacker")
    defensor = result.get("defender")
    etiqueta_celda = result.get("cell")

    if estado_resultado == "wrong_target":
        return [f"Turno de {atacante}, se esperaba ataque a {defensor}"]
    if estado_resultado == "invalid":
        return [f"Casilla {etiqueta_celda} ya usada, repite el disparo"]
    if estado_resultado == "agua":
        return [f"{atacante} dispara a {defensor} en {etiqueta_celda}: AGUA", "Cambio de turno"]
    if estado_resultado == "tocado":
        return [f"{atacante} dispara a {defensor} en {etiqueta_celda}: TOCADO", "Sigue el mismo turno"]
    if estado_resultado == "hundido":
        extra = "Fin de partida" if result.get("winner") else "Sigue el mismo turno"
        return [f"{atacante} hunde barco en {etiqueta_celda} de {defensor}", extra]
    if estado_resultado == "finished":
        ganador = result.get("winner")
        return [f"Partida terminada. Ganador: {ganador}"]
    return ["Ataque procesado"]


def _escribir_ultimo_resultado(result, messages, estado_partida):
    if result is None:
        return

    payload = {
        "timestamp": int(result.get("timestamp", 0) or time.time()),
        "messages": messages or [],
        "attacker": result.get("attacker"),
        "defender": result.get("defender"),
        "cell": result.get("cell"),
        "status": result.get("status"),
    }

    if estado_partida:
        payload["next_attacker"] = estado_partida.get("current_attacker")
        payload["next_defender"] = estado_partida.get("current_defender")
        payload["winner"] = estado_partida.get("winner")

    try:
        with open(LAST_RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except OSError as exc:  # noqa: PERF203
        print(f"[WARN] no se pudo escribir resultado en {LAST_RESULT_FILE}: {exc}")


def _capturar_estado_turno(estado_partida):
    return {
        "timestamp": int(time.time()),
        "attacker": estado_partida.get("current_attacker"),
        "defender": estado_partida.get("current_defender"),
        "status": "turn",
    }

if __name__ == "__main__":
    main()
