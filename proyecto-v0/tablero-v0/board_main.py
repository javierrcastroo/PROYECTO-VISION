# board_main.py
import cv2
import json
import os
import glob
import time
import numpy as np
from collections import Counter

CAPTURE_BUFFER_FRAMES = 150

from board_config import USE_UNDISTORT_BOARD, BOARD_CAMERA_PARAMS_PATH, WARP_SIZE
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
    if USE_UNDISTORT_BOARD and os.path.exists(BOARD_CAMERA_PARAMS_PATH):
        data = np.load(BOARD_CAMERA_PARAMS_PATH)
        mtx = data["camera_matrix"]
        dist = data["dist_coeffs"]
        

    # dos tableros
    boards_state_list = _init_boards()

    cv2.namedWindow("Tablero")
    cv2.setMouseCallback("Tablero", board_ui.board_mouse_callback)

    os.makedirs(ATTACKS_DIR, exist_ok=True)

    status = "STANDBY"  # STANDBY -> CAPTURING -> PLAYING
    capture_frames_left = 0
    accumulation = {"T1": [], "T2": []}
    stabilized_layouts = None
    game_state = None
    processed_attacks = set()
    last_validation_msgs = {}
    last_status_lines_printed = None
    status_lines = [
        "Standby: coloca barcos y calibra HSV",
        f"Pulsa 's' para fijar el layout ({CAPTURE_FRAMES} frames)",
    ]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # detectar el ORIGEN con ArUco cada frame
        aruco_utils.update_global_origin_from_aruco(frame, aruco_id=2)

        # procesar todos los tableros con el origen global actual
        vis, mask_b, mask_ship2, mask_ship1, mask_m, layouts = bp.process_all_boards(
            frame,
            boards_state_list,
            cam_mtx=mtx,
            dist=dist,
            max_boards=2,
            warp_size=WARP_SIZE,
        )

        validation_map = {}
        for layout in layouts:
            ok, msg = battleship_logic.evaluate_board(layout)
            validation_map[layout["name"]] = (ok, msg)

            if last_validation_msgs.get(layout["name"]) != msg:
                print(f"[{layout['name']}] {msg}")
                last_validation_msgs[layout["name"]] = msg

        if status == "CAPTURING":
            _accumulate_layouts(layouts, accumulation)
            capture_frames_left -= 1
            status_lines = [
                f"Capturando layout estable ({capture_frames_left} frames restantes)",
            ]
            if capture_frames_left <= 0:
                stabilized_layouts = _select_stable_layouts(accumulation, boards_state_list)
                game_state = battleship_logic.init_game_state(stabilized_layouts)
                status = "PLAYING"
                status_lines = [
                    "Layouts fijados. Empieza la partida.",
                    "Turno inicial: T1 ataca T2",
                ]
                _write_last_result(_snapshot_turn(game_state), status_lines, game_state)

        for slot in boards_state_list:
            if slot["name"] in validation_map and slot["last_quad"] is not None:
                ok, msg = validation_map[slot["name"]]
                board_ui.draw_validation_result(vis, slot["last_quad"], msg, ok)

        # dibujar el origen global si lo tenemos
        if board_state.GLOBAL_ORIGIN is not None:
            gx, gy = board_state.GLOBAL_ORIGIN
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

        if status == "PLAYING" and game_state is not None:
            if not game_state.get("finished"):
                status_lines = [
                    f"Turno de ataque: {game_state['current_attacker']} -> {game_state['current_defender']}",
                ]
            else:
                status_lines = [
                    f"Partida terminada. Ganador: {game_state.get('winner')}",
                ]

            pending_msg = _process_new_attacks(game_state, processed_attacks)
            if pending_msg:
                status_lines = pending_msg

            if game_state.get("finished") and _consume_restart_request():
                (
                    boards_state_list,
                    accumulation,
                    stabilized_layouts,
                    game_state,
                    processed_attacks,
                    status,
                    status_lines,
                    last_status_lines_printed,
                    capture_frames_left,
                ) = _restart_game()

        board_ui.draw_board_hud(vis)
        _draw_status_lines(vis, status_lines)

        if status_lines != last_status_lines_printed:
            for line in status_lines:
                print(line)
            last_status_lines_printed = list(status_lines)

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

        if key == ord("s") and status == "STANDBY":
            capture_frames_left = CAPTURE_FRAMES
            accumulation = {"T1": [], "T2": []}
            status = "CAPTURING"
            status_lines = [
                f"Inicio de captura de layout durante {CAPTURE_FRAMES} frames",
            ]
        elif key == ord("r"):
            (
                boards_state_list,
                accumulation,
                stabilized_layouts,
                game_state,
                processed_attacks,
                status,
                status_lines,
                last_status_lines_printed,
                capture_frames_left,
            ) = _restart_game()
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
        if bu.board_roi_defined:
            x0, x1 = sorted([bu.bx_start, bu.bx_end])
            y0, y1 = sorted([bu.by_start, bu.by_end])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = board_tracker.calibrate_board_color_from_roi(roi_hsv)
            board_tracker.current_lower, board_tracker.current_upper = lo, up
            print("[INFO] calibrado TABLERO:", lo, up)
        else:
            print("[WARN] dibuja ROI en 'Tablero' primero")

    elif key == ord("2"):
        if bu.board_roi_defined:
            x0, x1 = sorted([bu.bx_start, bu.bx_end])
            y0, y1 = sorted([bu.by_start, bu.by_end])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = object_tracker.calibrate_ship_two_color_from_roi(roi_hsv)
            object_tracker.current_ship_two_lower, object_tracker.current_ship_two_upper = lo, up
            print("[INFO] calibrado BARCO x2:", lo, up)
        else:
            print("[WARN] dibuja ROI sobre el barco largo")

    elif key == ord("1"):
        if bu.board_roi_defined:
            x0, x1 = sorted([bu.bx_start, bu.bx_end])
            y0, y1 = sorted([bu.by_start, bu.by_end])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = object_tracker.calibrate_ship_one_color_from_roi(roi_hsv)
            object_tracker.current_ship_one_lower, object_tracker.current_ship_one_upper = lo, up
            print("[INFO] calibrado BARCO x1:", lo, up)
        else:
            print("[WARN] dibuja ROI sobre el barco corto")

    elif key == ord("m"):
        if bu.board_roi_defined:
            x0, x1 = sorted([bu.bx_start, bu.bx_end])
            y0, y1 = sorted([bu.by_start, bu.by_end])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = object_tracker.calibrate_ammo_color_from_roi(roi_hsv)
            object_tracker.current_ammo_lower, object_tracker.current_ammo_upper = lo, up
            print("[INFO] calibrada MUNICION:", lo, up)
        else:
            print("[WARN] dibuja ROI sobre la municion")



def _init_boards():
    return [
        board_state.init_board_state("T1"),
        board_state.init_board_state("T2"),
    ]


def _snapshot_layout(layout):
    return {
        "ship_two_cells": tuple(sorted(layout.get("ship_two_cells", []))),
        "ship_one_cells": tuple(sorted(layout.get("ship_one_cells", []))),
        "board_size": layout.get("board_size", 5),
    }


def _accumulate_layouts(layouts, accumulation):
    for layout in layouts:
        name = layout.get("name")
        if name not in accumulation:
            accumulation[name] = []
        accumulation[name].append(_snapshot_layout(layout))


def _select_stable_layouts(samples_map, boards_state_list):
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


def _draw_status_lines(vis, lines):
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


def _process_new_attacks(game_state, processed_attacks):
    msgs = None
    files = sorted(glob.glob(os.path.join(ATTACKS_DIR, "T*_*.json")))
    for fp in files:
        fname = os.path.basename(fp)
        if fname in processed_attacks:
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] no se pudo leer {fname}: {exc}")
            processed_attacks.add(fname)
            continue

        target = payload.get("target")
        row = payload.get("row")
        col = payload.get("col")
        if target is None or row is None or col is None:
            print(f"[WARN] ataque {fname} incompleto")
            processed_attacks.add(fname)
            continue

        result = battleship_logic.apply_attack(game_state, target, row, col)
        processed_attacks.add(fname)

        msgs = _format_attack_result(result)
        _write_last_result(result, msgs, game_state)

        try:
            os.remove(fp)
        except OSError:
            pass
    return msgs


def _consume_restart_request():
    if not os.path.exists(RESTART_FILE):
        return False
    try:
        os.remove(RESTART_FILE)
    except OSError:
        pass
    return True


def _clear_pending_attack_files():
    try:
        for fp in glob.glob(os.path.join(ATTACKS_DIR, "T*_*.json")):
            os.remove(fp)
    except OSError:
        pass


def _restart_game():
    _reset_calibration_state()

    boards_state_list = _init_boards()
    accumulation = {"T1": [], "T2": []}
    stabilized_layouts = None
    game_state = None
    processed_attacks = set()
    board_state.GLOBAL_ORIGIN = None
    board_state.GLOBAL_ORIGIN_MISS = 0
    status = "STANDBY"
    status_lines = [
        "Reinicio solicitado. Coloca de nuevo los tableros y calibra si es necesario.",
        f"Pulsa 's' para fijar el layout ({CAPTURE_FRAMES} frames)",
    ]
    capture_frames_left = 0

    _clear_pending_attack_files()
    _write_last_result(
        {"timestamp": int(time.time()), "status": "reset"},
        status_lines,
        game_state,
    )
    last_status_lines_printed = None

    return (
        boards_state_list,
        accumulation,
        stabilized_layouts,
        game_state,
        processed_attacks,
        status,
        status_lines,
        last_status_lines_printed,
        capture_frames_left,
    )


def _reset_calibration_state():
    import board_tracker
    import object_tracker

    board_tracker.current_lower = board_tracker.DEFAULT_LOWER.copy()
    board_tracker.current_upper = board_tracker.DEFAULT_UPPER.copy()

    object_tracker.current_ship_two_lower = object_tracker.SHIP_TWO_LOWER_DEFAULT.copy()
    object_tracker.current_ship_two_upper = object_tracker.SHIP_TWO_UPPER_DEFAULT.copy()
    object_tracker.current_ship_one_lower = object_tracker.SHIP_ONE_LOWER_DEFAULT.copy()
    object_tracker.current_ship_one_upper = object_tracker.SHIP_ONE_UPPER_DEFAULT.copy()
    object_tracker.current_ammo_lower = object_tracker.AMMO_LOWER_DEFAULT.copy()
    object_tracker.current_ammo_upper = object_tracker.AMMO_UPPER_DEFAULT.copy()
    object_tracker.current_origin_lower = object_tracker.ORIG_LOWER_DEFAULT.copy()
    object_tracker.current_origin_upper = object_tracker.ORIG_UPPER_DEFAULT.copy()

    board_ui.board_roi_selecting = False
    board_ui.board_roi_defined = False
    board_ui.bx_start = board_ui.by_start = board_ui.bx_end = board_ui.by_end = 0
    board_ui.measure_points = []

    bp.clear_display_cache()


def _format_attack_result(result):
    if result is None:
        return None

    status = result.get("status")
    attacker = result.get("attacker")
    defender = result.get("defender")
    cell_label = result.get("cell")

    if status == "wrong_target":
        return [f"Turno de {attacker}, se esperaba ataque a {defender}"]
    if status == "invalid":
        return [f"Casilla {cell_label} ya usada, repite el disparo"]
    if status == "agua":
        return [f"{attacker} dispara a {defender} en {cell_label}: AGUA", "Cambio de turno"]
    if status == "tocado":
        return [f"{attacker} dispara a {defender} en {cell_label}: TOCADO", "Sigue el mismo turno"]
    if status == "hundido":
        extra = "Fin de partida" if result.get("winner") else "Sigue el mismo turno"
        return [f"{attacker} hunde barco en {cell_label} de {defender}", extra]
    if status == "finished":
        winner = result.get("winner")
        return [f"Partida terminada. Ganador: {winner}"]
    return ["Ataque procesado"]


def _write_last_result(result, messages, game_state):
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

    if game_state:
        payload["next_attacker"] = game_state.get("current_attacker")
        payload["next_defender"] = game_state.get("current_defender")
        payload["winner"] = game_state.get("winner")

    try:
        with open(LAST_RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except OSError as exc:  # noqa: PERF203
        print(f"[WARN] no se pudo escribir resultado en {LAST_RESULT_FILE}: {exc}")


def _snapshot_turn(game_state):
    return {
        "timestamp": int(time.time()),
        "attacker": game_state.get("current_attacker"),
        "defender": game_state.get("current_defender"),
        "status": "turn",
    }

if __name__ == "__main__":
    main()
