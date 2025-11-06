# main.py
import cv2
import os
import numpy as np

from config import (
    USE_UNDISTORT_HAND,
    USE_UNDISTORT_BOARD,
    HAND_CAMERA_PARAMS_PATH,
    BOARD_CAMERA_PARAMS_PATH,
)

import hand_pipeline
import board_pipeline


def main():
    # abrir cámaras
    cap_hand = cv2.VideoCapture(0)
    cap_board = cv2.VideoCapture(1)

    if not cap_hand.isOpened():
        raise RuntimeError("No se pudo abrir la cámara 0 (mano)")
    if not cap_board.isOpened():
        raise RuntimeError("No se pudo abrir la cámara 1 (tablero)")

    # cargar calibraciones
    HAND_CAM_MTX = HAND_DIST = None
    BOARD_CAM_MTX = BOARD_DIST = None

    if USE_UNDISTORT_HAND and os.path.exists(HAND_CAMERA_PARAMS_PATH):
        data = np.load(HAND_CAMERA_PARAMS_PATH)
        HAND_CAM_MTX = data["camera_matrix"]
        HAND_DIST = data["dist_coeffs"]
        print("[INFO] Undistort activado para cámara de la mano")

    if USE_UNDISTORT_BOARD and os.path.exists(BOARD_CAMERA_PARAMS_PATH):
        data = np.load(BOARD_CAMERA_PARAMS_PATH)
        BOARD_CAM_MTX = data["camera_matrix"]
        BOARD_DIST = data["dist_coeffs"]
        print("[INFO] Undistort activado para cámara del tablero")

    # estados de las dos tuberías
    hand_state = hand_pipeline.init_hand_state()
    board_state = board_pipeline.init_board_state()

    # ventanas y callbacks
    cv2.namedWindow("Mano")
    cv2.setMouseCallback("Mano", hand_state["mouse_cb"])

    cv2.namedWindow("Tablero")
    cv2.setMouseCallback("Tablero", board_state["mouse_cb"])

    while True:
        ok_hand, frame_hand = cap_hand.read()
        ok_board, frame_board = cap_board.read()
        if not ok_hand or not ok_board:
            break

        # ===== MANO =====
        vis_hand, mask_hand, skin_only, hsv_hand, feat_vec, hand_state = hand_pipeline.process_hand_frame(
            frame_hand, hand_state, HAND_CAM_MTX, HAND_DIST
        )

        # ===== TABLERO =====
        vis_board, masks_board, board_state = board_pipeline.process_board_frame(
            frame_board, board_state, BOARD_CAM_MTX, BOARD_DIST
        )

        # mostrar
        cv2.imshow("Mano", vis_hand)
        cv2.imshow("Mascara mano", mask_hand)
        cv2.imshow("Solo piel mano", skin_only)

        cv2.imshow("Tablero", vis_board)
        cv2.imshow("Mascara tablero", masks_board["board"])
        if masks_board["obj"] is not None:
            cv2.imshow("Mascara objetos", masks_board["obj"])
        if masks_board["orig"] is not None:
            cv2.imshow("Mascara origen", masks_board["orig"])

        # teclado
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

        # teclas de mano
        hand_state = hand_pipeline.handle_hand_key(
            key, hand_state, hsv_hand, feat_vec
        )

        # teclas de tablero
        board_state = board_pipeline.handle_board_key(
            key, board_state, frame_board
        )

    cap_hand.release()
    cap_board.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
