# board_main.py
import cv2
import os
import numpy as np

from board_config import (
    USE_UNDISTORT_BOARD,
    BOARD_CAMERA_PARAMS_PATH,
    WARP_SIZE,
)

import board_ui
import board_tracker
import object_tracker


def draw_quad(img, quad, color=(0, 255, 255)):
    if quad is None:
        return
    q = np.array(quad, dtype=np.int32)
    cv2.polylines(img, [q], True, color, 2)


def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara 1 (tablero)")

    BOARD_CAM_MTX = BOARD_DIST = None
    if USE_UNDISTORT_BOARD and os.path.exists(BOARD_CAMERA_PARAMS_PATH):
        data = np.load(BOARD_CAMERA_PARAMS_PATH)
        BOARD_CAM_MTX = data["camera_matrix"]
        BOARD_DIST = data["dist_coeffs"]
        print("[INFO] Undistort activado para la cámara del tablero")

    # estado
    last_board_quad = None
    last_ratio_cm_px = None
    last_height_px = None
    board_miss = 0
    MAX_BOARD_MISS = 10

    tracked_objs = {}
    next_obj_id = 1

    cv2.namedWindow("Tablero")
    cv2.setMouseCallback("Tablero", board_ui.board_mouse_callback)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # detectar tablero
        vis_board_detect, found_board, ratio_cm_px, height_px, mask_board, board_quad = board_tracker.detect_board(
            frame,
            camera_matrix=BOARD_CAM_MTX,
            dist_coeffs=BOARD_DIST
        )
        vis_board = vis_board_detect

        # fallback
        if found_board and board_quad is not None:
            last_board_quad = board_quad.copy()
            last_ratio_cm_px = ratio_cm_px
            last_height_px = height_px
            board_miss = 0
        else:
            board_miss += 1
            if last_board_quad is not None and board_miss <= MAX_BOARD_MISS:
                vis_board = frame.copy()
                draw_quad(vis_board, last_board_quad)
                cv2.putText(vis_board, "Tablero (fallback)", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                found_board = True
                board_quad = last_board_quad
                ratio_cm_px = last_ratio_cm_px
                height_px = last_height_px
            else:
                board_quad = None

        # HUD
        board_ui.draw_board_roi(vis_board)
        board_ui.draw_board_hud(vis_board)

        obj_mask = None
        origin_mask = None
        warp_img = np.zeros((WARP_SIZE, WARP_SIZE, 3), dtype=np.uint8)

        if found_board and board_quad is not None:
            hsv_board = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # aplanado
            src = np.array(board_quad, dtype=np.float32)
            dst = np.array([
                [0, 0],
                [WARP_SIZE - 1, 0],
                [WARP_SIZE - 1, WARP_SIZE - 1],
                [0, WARP_SIZE - 1]
            ], dtype=np.float32)
            H_warp = cv2.getPerspectiveTransform(src, dst)
            warp_img = cv2.warpPerspective(frame, H_warp, (WARP_SIZE, WARP_SIZE))

            # origen
            origin_centers, origin_mask = object_tracker.detect_colored_points_in_board(
                hsv_board,
                board_quad,
                object_tracker.current_origin_lower,
                object_tracker.current_origin_upper,
                max_objs=1,
                min_area=40
            )
            origin_point = origin_centers[0] if len(origin_centers) > 0 else None
            if origin_point is not None:
                cv2.circle(vis_board, origin_point, 7, (255, 255, 0), -1)
                cv2.putText(vis_board, "ORIG", (origin_point[0] + 5, origin_point[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            # objetos
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

            # tracking simple
            MAX_DIST = 35
            MAX_MISS = 10

            for oid in list(tracked_objs.keys()):
                tracked_objs[oid]['updated'] = False

            for (cx, cy) in obj_centers:
                best_oid = None
                best_dist = 1e9
                for oid, data in tracked_objs.items():
                    px, py = data['pt']
                    dist = ((cx - px)**2 + (cy - py)**2)**0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_oid = oid
                if best_oid is not None and best_dist < MAX_DIST:
                    tracked_objs[best_oid]['pt'] = (cx, cy)
                    tracked_objs[best_oid]['miss'] = 0
                    tracked_objs[best_oid]['updated'] = True
                else:
                    tracked_objs[next_obj_id] = {
                        'pt': (cx, cy),
                        'miss': 0,
                        'updated': True,
                    }
                    next_obj_id += 1

            for oid in list(tracked_objs.keys()):
                if not tracked_objs[oid].get('updated', False):
                    tracked_objs[oid]['miss'] += 1
                if tracked_objs[oid]['miss'] > MAX_MISS:
                    del tracked_objs[oid]

            # homografía + casillas (con Y invertida como vimos)
            if origin_point is not None and len(tracked_objs) > 0:
                board_w = board_tracker.BOARD_SQUARES * board_tracker.SQUARE_SIZE_CM
                board_h = board_tracker.BOARD_SQUARES * board_tracker.SQUARE_SIZE_CM
                src_h = np.array(board_quad, dtype=np.float32)
                dst_h = np.array([
                    [0, 0],
                    [board_w, 0],
                    [board_w, board_h],
                    [0, board_h],
                ], dtype=np.float32)
                H = cv2.getPerspectiveTransform(src_h, dst_h)

                def warp_pt(pt):
                    p = np.array([[pt]], dtype=np.float32)
                    pw = cv2.perspectiveTransform(p, H)
                    return pw[0, 0, 0], pw[0, 0, 1]

                origin_x, origin_y = warp_pt(origin_point)
                cell_size = board_tracker.SQUARE_SIZE_CM
                max_cells = board_tracker.BOARD_SQUARES

                y_off = 120
                for oid, data in tracked_objs.items():
                    ox_img, oy_img = data['pt']
                    ox_world, oy_world = warp_pt((ox_img, oy_img))

                    rel_x = ox_world - origin_x
                    rel_y = origin_y - oy_world  # invertido

                    if rel_x < 0: rel_x = 0
                    if rel_y < 0: rel_y = 0

                    col = int(rel_x // cell_size)
                    row = int(rel_y // cell_size)
                    col = max(0, min(max_cells - 1, col))
                    row = max(0, min(max_cells - 1, row))

                    col_name = chr(ord('A') + col)
                    row_name = row + 1
                    cell_label = f"{col_name}{row_name}"

                    cv2.putText(vis_board,
                                f"O{oid}: {cell_label} ({rel_x:.1f},{rel_y:.1f})cm",
                                (10, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    y_off += 15

        else:
            # no hay tablero
            for oid in list(tracked_objs.keys()):
                tracked_objs[oid]['miss'] += 1
                if tracked_objs[oid]['miss'] > 15:
                    del tracked_objs[oid]

        # mostrar
        cv2.imshow("Tablero", vis_board)
        cv2.imshow("Mascara tablero", mask_board)
        cv2.imshow("Tablero aplanado", warp_img)
        if obj_mask is not None:
            cv2.imshow("Mascara objetos", obj_mask)
        if origin_mask is not None:
            cv2.imshow("Mascara origen", origin_mask)

        # teclado
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

        # calibraciones propias del tablero
        if key == ord('b'):
            if board_ui.board_roi_defined:
                x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
                y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
                roi_bgr = frame[y0:y1, x0:x1]
                hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                lower, upper = board_tracker.calibrate_board_color_from_roi(hsv_roi)
                board_tracker.current_lower = lower
                board_tracker.current_upper = upper
                print("[INFO] calibrado TABLERO:", lower, upper)
            else:
                print("[WARN] dibuja ROI en 'Tablero'")

        elif key == ord('o'):
            if board_ui.board_roi_defined:
                x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
                y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
                roi_bgr = frame[y0:y1, x0:x1]
                hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                lo, up = object_tracker.calibrate_object_color_from_roi(hsv_roi)
                object_tracker.current_obj_lower = lo
                object_tracker.current_obj_upper = up
                print("[INFO] calibrado OBJETO:", lo, up)
            else:
                print("[WARN] dibuja ROI en 'Tablero' sobre la ficha")

        elif key == ord('r'):
            if board_ui.board_roi_defined:
                x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
                y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
                roi_bgr = frame[y0:y1, x0:x1]
                hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                lo, up = object_tracker.calibrate_origin_color_from_roi(hsv_roi)
                object_tracker.current_origin_lower = lo
                object_tracker.current_origin_upper = up
                print("[INFO] calibrado ORIGEN:", lo, up)
            else:
                print("[WARN] dibuja ROI en 'Tablero' sobre el origen")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
