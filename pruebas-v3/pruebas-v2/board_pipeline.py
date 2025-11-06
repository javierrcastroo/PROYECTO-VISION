# board_pipeline.py
import cv2
import numpy as np

import board_ui
import board_tracker
import object_tracker

WARP_SIZE = 500  # ventana del tablero aplanado


def init_board_state():
    return {
        "mouse_cb": board_ui.board_mouse_callback,
        "last_board_quad": None,
        "last_ratio_cm_px": None,
        "last_height_px": None,
        "board_miss": 0,
        "tracked_objs": {},   # id -> {'pt':(x,y), 'miss':int}
        "next_obj_id": 1,
    }


def process_board_frame(frame_bgr, state, cam_mtx=None, dist=None):
    # 1. detectar tablero (como antes)
    vis_detect, found, ratio_cm_px, height_px, mask_board, board_quad = board_tracker.detect_board(
        frame_bgr,
        camera_matrix=cam_mtx,
        dist_coeffs=dist
    )
    vis_board = vis_detect

    # 2. fallback
    if found and board_quad is not None:
        state["last_board_quad"] = board_quad.copy()
        state["last_ratio_cm_px"] = ratio_cm_px
        state["last_height_px"] = height_px
        state["board_miss"] = 0
    else:
        state["board_miss"] += 1
        if state["last_board_quad"] is not None and state["board_miss"] <= 10:
            vis_board = frame_bgr.copy()
            _draw_quad(vis_board, state["last_board_quad"])
            cv2.putText(vis_board, "Tablero (fallback)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            found = True
            board_quad = state["last_board_quad"]
            ratio_cm_px = state["last_ratio_cm_px"]
            height_px = state["last_height_px"]
        else:
            board_quad = None

    # HUD
    board_ui.draw_board_roi(vis_board)
    board_ui.draw_board_hud(vis_board)

    obj_mask = None
    origin_mask = None
    warp_img = np.zeros((WARP_SIZE, WARP_SIZE, 3), dtype=np.uint8)

    if found and board_quad is not None:
        hsv_board = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # 2.1 aplanado
        src = np.array(board_quad, dtype=np.float32)
        dst = np.array([
            [0, 0],
            [WARP_SIZE - 1, 0],
            [WARP_SIZE - 1, WARP_SIZE - 1],
            [0, WARP_SIZE - 1]
        ], dtype=np.float32)
        H_warp = cv2.getPerspectiveTransform(src, dst)
        warp_img = cv2.warpPerspective(frame_bgr, H_warp, (WARP_SIZE, WARP_SIZE))

        # 3. origen
        origin_centers, origin_mask = object_tracker.detect_colored_points_in_board(
            hsv_board,
            board_quad,
            object_tracker.current_origin_lower,
            object_tracker.current_origin_upper,
            max_objs=1,
            min_area=40
        )
        origin_pt = origin_centers[0] if len(origin_centers) > 0 else None
        if origin_pt is not None:
            cv2.circle(vis_board, origin_pt, 7, (255, 255, 0), -1)
            cv2.putText(vis_board, "ORIG", (origin_pt[0] + 5, origin_pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # 4. objetos
        obj_centers, obj_mask = object_tracker.detect_colored_points_in_board(
            hsv_board,
            board_quad,
            object_tracker.current_obj_lower,
            object_tracker.current_obj_upper,
            max_objs=4,
            min_area=40
        )
        for i, (cx, cy) in enumerate(obj_centers):
            cv2.circle(vis_board, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(vis_board, f"O{i+1}", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # 5. tracking
        state["tracked_objs"], state["next_obj_id"] = _update_tracks(
            state["tracked_objs"], obj_centers, state["next_obj_id"]
        )

        # 6. homografía + casillas
        if origin_pt is not None and len(state["tracked_objs"]) > 0:
            _project_and_label(
                vis_board,
                board_quad,
                origin_pt,
                state["tracked_objs"]
            )
    else:
        # no hay tablero -> limpiar tracking poco a poco
        for oid in list(state["tracked_objs"].keys()):
            state["tracked_objs"][oid]['miss'] += 1
            if state["tracked_objs"][oid]['miss'] > 15:
                del state["tracked_objs"][oid]

    masks = {
        "board": mask_board,
        "obj": obj_mask,
        "orig": origin_mask,
        "warp": warp_img,
    }
    return vis_board, masks, state


def handle_board_key(key, state, frame_board):
    # calibrar tablero
    if key == ord('b'):
        if board_ui.board_roi_defined:
            x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
            y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
            roi_bgr = frame_board[y0:y1, x0:x1]
            hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            lower, upper = board_tracker.calibrate_board_color_from_roi(hsv_roi)
            board_tracker.current_lower = lower
            board_tracker.current_upper = upper
            print("[INFO] calibrado color TABLERO:", lower, upper)
        else:
            print("[WARN] dibuja un ROI en 'Tablero' antes de pulsar b")

    # calibrar objeto
    elif key == ord('o'):
        if board_ui.board_roi_defined:
            x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
            y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
            roi_bgr = frame_board[y0:y1, x0:x1]
            hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            lo, up = object_tracker.calibrate_object_color_from_roi(hsv_roi)
            object_tracker.current_obj_lower = lo
            object_tracker.current_obj_upper = up
            print("[INFO] calibrado color OBJETO:", lo, up)
        else:
            print("[WARN] dibuja un ROI en 'Tablero' sobre el objeto")

    # calibrar origen
    elif key == ord('r'):
        if board_ui.board_roi_defined:
            x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
            y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
            roi_bgr = frame_board[y0:y1, x0:x1]
            hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            lo, up = object_tracker.calibrate_origin_color_from_roi(hsv_roi)
            object_tracker.current_origin_lower = lo
            object_tracker.current_origin_upper = up
            print("[INFO] calibrado color ORIGEN:", lo, up)
        else:
            print("[WARN] dibuja un ROI en 'Tablero' sobre el origen")

    return state


# ===== helpers =====

def _draw_quad(img, quad, color=(0, 255, 255)):
    q = np.array(quad, dtype=np.int32)
    cv2.polylines(img, [q], True, color, 2)


def _update_tracks(tracked, detections, next_id, max_dist=35, max_miss=10):
    for oid in list(tracked.keys()):
        tracked[oid]['updated'] = False

    for (cx, cy) in detections:
        best_oid = None
        best_dist = 1e9
        for oid, data in tracked.items():
            px, py = data['pt']
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_oid = oid
        if best_oid is not None and best_dist < max_dist:
            tracked[best_oid]['pt'] = (cx, cy)
            tracked[best_oid]['miss'] = 0
            tracked[best_oid]['updated'] = True
        else:
            tracked[next_id] = {
                'pt': (cx, cy),
                'miss': 0,
                'updated': True,
            }
            next_id += 1

    for oid in list(tracked.keys()):
        if not tracked[oid].get('updated', False):
            tracked[oid]['miss'] += 1
        if tracked[oid]['miss'] > max_miss:
            del tracked[oid]

    return tracked, next_id


def _project_and_label(vis_board, board_quad, origin_pt, tracked_objs):
    board_w = board_tracker.BOARD_SQUARES * board_tracker.SQUARE_SIZE_CM
    board_h = board_tracker.BOARD_SQUARES * board_tracker.SQUARE_SIZE_CM
    src = np.array(board_quad, dtype=np.float32)
    dst = np.array([
        [0, 0],
        [board_w, 0],
        [board_w, board_h],
        [0, board_h],
    ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)

    def warp(pt):
        p = np.array([[pt]], dtype=np.float32)
        pw = cv2.perspectiveTransform(p, H)
        return pw[0, 0, 0], pw[0, 0, 1]

    origin_x, origin_y = warp(origin_pt)

    cell_size = board_tracker.SQUARE_SIZE_CM
    n_cells = board_tracker.BOARD_SQUARES

    y_offset = 120
    line_h = 15

    for oid, data in tracked_objs.items():
        ox_img, oy_img = data["pt"]
        ox_world, oy_world = warp((ox_img, oy_img))

        rel_x = ox_world - origin_x
        # Y invertida porque el origen está abajo
        rel_y = origin_y - oy_world
        if rel_x < 0: rel_x = 0
        if rel_y < 0: rel_y = 0

        col = int(rel_x // cell_size)
        row = int(rel_y // cell_size)

        col = max(0, min(n_cells - 1, col))
        row = max(0, min(n_cells - 1, row))

        col_name = chr(ord('A') + col)
        row_name = row + 1
        cell_label = f"{col_name}{row_name}"

        center_x = col * cell_size + cell_size / 2
        center_y = row * cell_size + cell_size / 2

        cv2.putText(
            vis_board,
            f"O{oid}: {cell_label} ({rel_x:.1f},{rel_y:.1f})cm",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )
        y_offset += line_h

        print(f"[O{oid}] celda={cell_label} | rel=({rel_x:.2f},{rel_y:.2f}) cm | centro=({center_x:.2f},{center_y:.2f}) cm")
