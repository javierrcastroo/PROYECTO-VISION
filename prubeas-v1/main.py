import cv2
import numpy as np
import os
from collections import deque, Counter

from config import *
from segmentation import calibrate_from_roi, segment_hand_mask
from features import compute_feature_vector
from storage import save_gesture_example, load_gesture_gallery, save_sequence_json
from classifier import knn_predict
import ui
import board_ui          # <--- nuevo
import board_tracker     # <--- nuevo

def majority_vote(labels):
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]

def main():
    cap_hand = cv2.VideoCapture(0)
    cap_board = cv2.VideoCapture(1)

    if not cap_hand.isOpened() or not cap_board.isOpened():
        raise RuntimeError("No se pudieron abrir las cámaras")

    # cargar params
    HAND_CAM_MTX = HAND_DIST = None
    BOARD_CAM_MTX = BOARD_DIST = None

    if USE_UNDISTORT_HAND and os.path.exists(HAND_CAMERA_PARAMS_PATH):
        p = np.load(HAND_CAMERA_PARAMS_PATH)
        HAND_CAM_MTX = p["camera_matrix"]; HAND_DIST = p["dist_coeffs"]

    if USE_UNDISTORT_BOARD and os.path.exists(BOARD_CAMERA_PARAMS_PATH):
        p = np.load(BOARD_CAMERA_PARAMS_PATH)
        BOARD_CAM_MTX = p["camera_matrix"]; BOARD_DIST = p["dist_coeffs"]

    cv2.namedWindow("Mano")
    cv2.setMouseCallback("Mano", ui.mouse_callback)

    cv2.namedWindow("Tablero")
    cv2.setMouseCallback("Tablero", board_ui.board_mouse_callback)

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

        # ===== mano =====
        if HAND_CAM_MTX is not None:
            frame_hand = cv2.undistort(frame_hand, HAND_CAM_MTX, HAND_DIST)
        frame_hand = cv2.flip(frame_hand, 1)
        frame_hand = cv2.resize(frame_hand, (PREVIEW_W, PREVIEW_H))
        vis_hand = frame_hand.copy()
        hsv_hand = cv2.cvtColor(frame_hand, cv2.COLOR_BGR2HSV)

        ui.draw_roi_rectangle(vis_hand)
        mask_largest = segment_hand_mask(hsv_hand, lower_skin, upper_skin)
        ui.draw_hand_box(vis_hand, mask_largest)
        skin_only = cv2.bitwise_and(frame_hand, frame_hand, mask=mask_largest)

        feat_vec = compute_feature_vector(mask_largest)
        best_dist = None
        per_frame_label = None
        if feat_vec is not None and RECOGNIZE_MODE:
            raw_label, best_dist = knn_predict(feat_vec, gallery, k=5)
            if raw_label is not None and best_dist is not None:
                per_frame_label = raw_label if best_dist <= CONFIDENCE_THRESHOLD else "????"
        if per_frame_label is not None:
            recent_preds.append(per_frame_label)
        stable_label = majority_vote(list(recent_preds))

        ui.draw_hud(vis_hand, lower_skin, upper_skin, current_label)
        ui.draw_prediction(vis_hand, stable_label, best_dist or 0.0)

        cv2.imshow("Mano", vis_hand)
        cv2.imshow("Mascara mano", mask_largest)
        cv2.imshow("Solo piel mano", skin_only)
        # ===== tablero =====
        vis_board, found_board, ratio_cm_px, height_px, mask_board, board_quad = board_tracker.detect_board(
            frame_board,
            camera_matrix=BOARD_CAM_MTX,
            dist_coeffs=BOARD_DIST
        )

        # dibujar ROI del tablero (clic izq)
        board_ui.draw_board_roi(vis_board)

        # --- detección automática de objetos ---
        obj_centers = []
        obj_mask = None

        if found_board and board_quad is not None:
            # necesitamos también el HSV del frame del tablero
            hsv_board = cv2.cvtColor(frame_board, cv2.COLOR_BGR2HSV)

            # detectar objetos del color calibrado
            import object_tracker
            obj_centers, obj_mask = object_tracker.detect_objects_in_board(
                frame_board,
                hsv_board,
                board_quad,
                max_objs=2
            )

            # dibujar los objetos encontrados
            for (cx, cy) in obj_centers:
                cv2.circle(vis_board, (cx, cy), 7, (0,0,255), -1)

            # si tenemos 2 objetos y tenemos ratio, medimos
            if len(obj_centers) == 2 and ratio_cm_px is not None:
                (x1, y1), (x2, y2) = obj_centers
                dist_px = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
                dist_cm = dist_px * ratio_cm_px

                cv2.line(vis_board, (x1, y1), (x2, y2), (255, 0, 255), 2)
                mx, my = int((x1+x2)/2), int((y1+y2)/2)
                cv2.putText(vis_board, f"{dist_cm:.1f} cm",
                            (mx+5, my-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255,0,255), 2, cv2.LINE_AA)

        # mostrar ventanas
        cv2.imshow("Tablero", vis_board)
        cv2.imshow("Mascara tablero", mask_board)
        if obj_mask is not None:
            cv2.imshow("Mascara objetos", obj_mask)




        # ===== teclado =====
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # calibrar mano
            if ui.roi_defined:
                x0, x1 = sorted([ui.x_start, ui.x_end])
                y0, y1 = sorted([ui.y_start, ui.y_end])
                if (x1-x0)>5 and (y1-y0)>5:
                    roi_hsv = hsv_hand[y0:y1, x0:x1]
                    lower_skin, upper_skin = calibrate_from_roi(roi_hsv)
                    print("[INFO] calibrado hsv mano:", lower_skin, upper_skin)
        elif key == ord('b'):
            # calibrar tablero desde ROI de tablero
            if board_ui.board_roi_defined:
                x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
                y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
                roi_bgr = frame_board[y0:y1, x0:x1]
                hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                lower, upper = board_tracker.calibrate_board_color_from_roi(hsv_roi)
                board_tracker.current_lower = lower
                board_tracker.current_upper = upper
                print("[INFO] calibrado color tablero:", lower, upper)
            else:
                print("[WARN] dibuja primero un ROI en 'Tablero'")
        elif key == ord('g'):
            if lower_skin is None:
                print("[WARN] no hay hsv mano")
            else:
                if feat_vec is None:
                    print("[WARN] mano no válida")
                else:
                    save_gesture_example(feat_vec, current_label)
                    gallery.append((feat_vec, current_label))
        elif key == ord('o'):
            # calibrar color de OBJETO desde el ROI del tablero
            if board_ui.board_roi_defined:
                x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
                y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
                roi_bgr = frame_board[y0:y1, x0:x1]
                hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                import object_tracker
                lower_o, upper_o = object_tracker.calibrate_object_color_from_roi(hsv_roi)
                object_tracker.current_obj_lower = lower_o
                object_tracker.current_obj_upper = upper_o
                print("[INFO] calibrado color OBJETO:", lower_o, upper_o)
            else:
                print("[WARN] dibuja primero un ROI en 'Tablero' sobre el objeto")

        elif key == ord('a'):
            acciones = ui.append_action(acciones, stable_label)
        elif key == ord('p'):
            save_sequence_json(acciones); acciones.clear()
        elif key == ord('0'): current_label = "0dedos"
        elif key == ord('1'): current_label = "1dedo"
        elif key == ord('2'): current_label = "2dedos"
        elif key == ord('3'): current_label = "3dedos"
        elif key == ord('4'): current_label = "4dedos"
        elif key == ord('5'): current_label = "5dedos"
        elif key == ord('q') or key == 27:
            break

    cap_hand.release()
    cap_board.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
