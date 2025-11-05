# main.py
import cv2
import numpy as np
import os
from collections import deque, Counter

from config import (
    PREVIEW_W, PREVIEW_H,
    RECOGNIZE_MODE,
    CONFIDENCE_THRESHOLD,
    USE_UNDISTORT_HAND,
    USE_UNDISTORT_BOARD,
    HAND_CAMERA_PARAMS_PATH,
    BOARD_CAMERA_PARAMS_PATH,
)

from segmentation import calibrate_from_roi, segment_hand_mask
from features import compute_feature_vector
from storage import save_gesture_example, load_gesture_gallery, save_sequence_json
from classifier import knn_predict
import ui
import board_ui
import board_tracker
import object_tracker


def majority_vote(labels):
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def main():
    # abrir cámaras
    cap_hand = cv2.VideoCapture(0)
    cap_board = cv2.VideoCapture(1)

    if not cap_hand.isOpened():
        raise RuntimeError("No se pudo abrir cámara mano (0)")
    if not cap_board.isOpened():
        raise RuntimeError("No se pudo abrir cámara tablero (1)")

    # cargar parámetros de cámara si hay
    HAND_CAM_MTX = HAND_DIST = None
    BOARD_CAM_MTX = BOARD_DIST = None

    if USE_UNDISTORT_HAND and os.path.exists(HAND_CAMERA_PARAMS_PATH):
        p = np.load(HAND_CAMERA_PARAMS_PATH)
        HAND_CAM_MTX = p["camera_matrix"]
        HAND_DIST = p["dist_coeffs"]
        print("[INFO] Undistort activado para cámara de la mano (0).")

    if USE_UNDISTORT_BOARD and os.path.exists(BOARD_CAMERA_PARAMS_PATH):
        p = np.load(BOARD_CAMERA_PARAMS_PATH)
        BOARD_CAM_MTX = p["camera_matrix"]
        BOARD_DIST = p["dist_coeffs"]
        print("[INFO] Undistort activado para cámara del tablero (1).")

    # ventanas y callbacks
    cv2.namedWindow("Mano")
    cv2.setMouseCallback("Mano", ui.mouse_callback)

    cv2.namedWindow("Tablero")
    cv2.setMouseCallback("Tablero", board_ui.board_mouse_callback)

    # estado de mano
    lower_skin = upper_skin = None
    gallery = load_gesture_gallery() if RECOGNIZE_MODE else []
    acciones = []
    current_label = "2dedos"
    recent_preds = deque(maxlen=7)

    while True:
        ret_hand, frame_hand = cap_hand.read()
        ret_board, frame_board = cap_board.read()
        if not ret_hand or not ret_board:
            break

        # ===================== MANO (cam0) =====================
        if HAND_CAM_MTX is not None:
            frame_hand = cv2.undistort(frame_hand, HAND_CAM_MTX, HAND_DIST)

        frame_hand = cv2.flip(frame_hand, 1)
        frame_hand = cv2.resize(frame_hand, (PREVIEW_W, PREVIEW_H))
        vis_hand = frame_hand.copy()
        hsv_hand = cv2.cvtColor(frame_hand, cv2.COLOR_BGR2HSV)

        # ROI de la mano
        ui.draw_roi_rectangle(vis_hand)

        # segmentar mano
        mask_largest = segment_hand_mask(hsv_hand, lower_skin, upper_skin)
        ui.draw_hand_box(vis_hand, mask_largest)
        skin_only = cv2.bitwise_and(frame_hand, frame_hand, mask=mask_largest)

        # features y predicción
        best_dist = None
        per_frame_label = None
        feat_vec = compute_feature_vector(mask_largest)
        if feat_vec is not None and RECOGNIZE_MODE:
            raw_label, best_dist = knn_predict(feat_vec, gallery, k=5)
            if raw_label is not None and best_dist is not None:
                per_frame_label = raw_label if best_dist <= CONFIDENCE_THRESHOLD else "????"

        if per_frame_label is not None:
            recent_preds.append(per_frame_label)
        stable_label = majority_vote(list(recent_preds))

        ui.draw_hud(vis_hand, lower_skin, upper_skin, current_label)
        ui.draw_prediction(vis_hand, stable_label, best_dist if best_dist is not None else 0.0)

        cv2.imshow("Mano", vis_hand)
        cv2.imshow("Mascara mano", mask_largest)
        cv2.imshow("Solo piel mano", skin_only)

        # ===================== TABLERO (cam1) =====================
        # detectar tablero por color (ya devuelve quad)
        vis_board, found_board, ratio_cm_px, height_px, mask_board, board_quad = board_tracker.detect_board(
            frame_board,
            camera_matrix=BOARD_CAM_MTX,
            dist_coeffs=BOARD_DIST
        )

        # dibujar ROI del tablero
        board_ui.draw_board_roi(vis_board)
        board_ui.draw_board_hud(vis_board)

        obj_mask = None
        origin_mask = None

        # si hay tablero y su cuadrilátero
        if found_board and board_quad is not None:
            # necesitamos HSV del tablero para detectar objetos/origen
            hsv_board = cv2.cvtColor(frame_board, cv2.COLOR_BGR2HSV)

            # 1. detectar origen (color calibrado con 'r')
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
                cv2.putText(vis_board, "ORIG", (origin_point[0]+5, origin_point[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)

            # 2. detectar objetos (color calibrado con 'o')
            obj_centers, obj_mask = object_tracker.detect_colored_points_in_board(
                hsv_board,
                board_quad,
                object_tracker.current_obj_lower,
                object_tracker.current_obj_upper,
                max_objs=4,
                min_area=40
            )
            for i, (cx, cy) in enumerate(obj_centers):
                cv2.circle(vis_board, (cx, cy), 6, (0,0,255), -1)
                cv2.putText(vis_board, f"O{i+1}", (cx+5, cy-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)

            # 3. si tenemos origen y objetos -> pasar a coords tablero
            if origin_point is not None and len(obj_centers) > 0:
                # tamaño real del tablero en cm
                board_w = board_tracker.BOARD_SQUARES * board_tracker.SQUARE_SIZE_CM
                board_h = board_tracker.BOARD_SQUARES * board_tracker.SQUARE_SIZE_CM

                src = np.array(board_quad, dtype=np.float32)
                dst = np.array([
                    [0, 0],
                    [board_w, 0],
                    [board_w, board_h],
                    [0, board_h]
                ], dtype=np.float32)

                H = cv2.getPerspectiveTransform(src, dst)

                def warp_pt(pt):
                    p = np.array([[pt]], dtype=np.float32)
                    pw = cv2.perspectiveTransform(p, H)
                    return pw[0,0,0], pw[0,0,1]

                # origen en sistema tablero
                origin_x, origin_y = warp_pt(origin_point)

                # para cada objeto, coordenadas relativas al origen
                y_offset = 15
                for i, (cx, cy) in enumerate(obj_centers):
                    ox, oy = warp_pt((cx, cy))
                    rel_x = ox - origin_x
                    rel_y = oy - origin_y

                    # lo pintamos en la imagen
                    cv2.putText(vis_board,
                                f"O{i+1}: ({rel_x:.1f}, {rel_y:.1f}) cm",
                                (10, 120 + i*y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0,255,255), 1)

                    # y también por consola
                    print(f"[OBJ{i+1}] X={rel_x:.2f} cm, Y={rel_y:.2f} cm")

        # mostrar ventanas de tablero
        cv2.imshow("Tablero", vis_board)
        cv2.imshow("Mascara tablero", mask_board)
        if obj_mask is not None:
            cv2.imshow("Mascara objetos", obj_mask)
        if origin_mask is not None:
            cv2.imshow("Mascara origen", origin_mask)

        # ===================== TECLADO =====================
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # calibrar piel en la cámara de la mano
            if ui.roi_defined:
                x0, x1 = sorted([ui.x_start, ui.x_end])
                y0, y1 = sorted([ui.y_start, ui.y_end])
                if (x1 - x0) > 5 and (y1 - y0) > 5:
                    roi_hsv = hsv_hand[y0:y1, x0:x1]
                    lower_skin, upper_skin = calibrate_from_roi(roi_hsv)
                    print("[INFO] calibrado hsv mano:", lower_skin, upper_skin)
                else:
                    print("[WARN] ROI de mano demasiado pequeño.")
            else:
                print("[WARN] dibuja primero un ROI en 'Mano'.")

        elif key == ord('b'):
            # calibrar color del TABLERO desde el ROI de la ventana Tablero
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
                print("[WARN] dibuja primero un ROI en 'Tablero' sobre el tablero")

        elif key == ord('o'):
            # calibrar color de OBJETO desde ROI del tablero
            if board_ui.board_roi_defined:
                x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
                y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
                roi_bgr = frame_board[y0:y1, x0:x1]
                hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                lower_o, upper_o = object_tracker.calibrate_object_color_from_roi(hsv_roi)
                object_tracker.current_obj_lower = lower_o
                object_tracker.current_obj_upper = upper_o
                print("[INFO] calibrado color OBJETO:", lower_o, upper_o)
            else:
                print("[WARN] dibuja primero un ROI en 'Tablero' sobre el objeto")

        elif key == ord('r'):
            # calibrar color de ORIGEN desde ROI del tablero
            if board_ui.board_roi_defined:
                x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
                y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
                roi_bgr = frame_board[y0:y1, x0:x1]
                hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                lower_o, upper_o = object_tracker.calibrate_origin_color_from_roi(hsv_roi)
                object_tracker.current_origin_lower = lower_o
                object_tracker.current_origin_upper = upper_o
                print("[INFO] calibrado color ORIGEN:", lower_o, upper_o)
            else:
                print("[WARN] dibuja primero un ROI en 'Tablero' sobre el origen")

        elif key == ord('g'):
            # guardar gesto de la mano
            if lower_skin is None or upper_skin is None:
                print("[WARN] no hay calibración HSV de mano. No guardo.")
            else:
                if feat_vec is None:
                    print("[WARN] no se detecta mano válida.")
                else:
                    save_gesture_example(feat_vec, current_label)
                    if RECOGNIZE_MODE:
                        gallery.append((feat_vec, current_label))
                    print(f"[INFO] guardado gesto con label={current_label}")

        elif key == ord('a'):
            acciones = ui.append_action(acciones, stable_label)

        elif key == ord('p'):
            save_sequence_json(acciones)
            print(f"[INFO] Lista gestos: {acciones}")
            acciones.clear()

        elif key == ord('0'):
            current_label = "0dedos"
        elif key == ord('1'):
            current_label = "1dedo"
        elif key == ord('2'):
            current_label = "2dedos"
        elif key == ord('3'):
            current_label = "3dedos"
        elif key == ord('4'):
            current_label = "4dedos"
        elif key == ord('5'):
            current_label = "5dedos"

        elif key == ord('q') or key == 27:
            break

    cap_hand.release()
    cap_board.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
