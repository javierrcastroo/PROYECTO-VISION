# hand_pipeline.py
import cv2
from collections import deque

from config import PREVIEW_W, PREVIEW_H, RECOGNIZE_MODE, CONFIDENCE_THRESHOLD
from segmentation import segment_hand_mask, calibrate_from_roi
from features import compute_feature_vector
from classifier import knn_predict
from storage import save_gesture_example, load_gesture_gallery, save_sequence_json
import ui


def init_hand_state():
    return {
        "lower_skin": None,
        "upper_skin": None,
        "current_label": "2dedos",
        "recent_preds": deque(maxlen=7),
        "gallery": load_gesture_gallery() if RECOGNIZE_MODE else [],
        "acciones": [],
        "mouse_cb": ui.mouse_callback,
        "stable_label": None,
    }


def process_hand_frame(frame_bgr, state, cam_mtx=None, dist=None):
    # undistort
    if cam_mtx is not None:
        frame_bgr = cv2.undistort(frame_bgr, cam_mtx, dist)

    # espejo y resize
    frame_bgr = cv2.flip(frame_bgr, 1)
    frame_bgr = cv2.resize(frame_bgr, (PREVIEW_W, PREVIEW_H))
    vis = frame_bgr.copy()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # ROI mano
    ui.draw_roi_rectangle(vis)

    # segmentar mano
    mask = segment_hand_mask(hsv, state["lower_skin"], state["upper_skin"])
    ui.draw_hand_box(vis, mask)
    skin_only = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

    # features
    feat_vec = compute_feature_vector(mask)

    # predicción opcional
    per_frame_label = None
    best_dist = None
    if feat_vec is not None and RECOGNIZE_MODE:
        raw_label, best_dist = knn_predict(feat_vec, state["gallery"], k=5)
        if raw_label is not None and best_dist is not None:
            per_frame_label = raw_label if best_dist <= CONFIDENCE_THRESHOLD else "????"

    if per_frame_label is not None:
        state["recent_preds"].append(per_frame_label)

    # voto estable
    if state["recent_preds"]:
        state["stable_label"] = max(
            set(state["recent_preds"]),
            key=state["recent_preds"].count
        )
    else:
        state["stable_label"] = None

    # HUD mano
    ui.draw_hud(vis, state["lower_skin"], state["upper_skin"], state["current_label"])
    ui.draw_prediction(vis, state["stable_label"], best_dist if best_dist else 0.0)

    return vis, mask, skin_only, hsv, feat_vec, state


def handle_hand_key(key, state, hsv_hand, feat_vec):
    # calibrar piel
    if key == ord('c'):
        if ui.roi_defined:
            x0, x1 = sorted([ui.x_start, ui.x_end])
            y0, y1 = sorted([ui.y_start, ui.y_end])
            if (x1 - x0) > 5 and (y1 - y0) > 5:
                roi_hsv = hsv_hand[y0:y1, x0:x1]
                lower, upper = calibrate_from_roi(roi_hsv)
                state["lower_skin"] = lower
                state["upper_skin"] = upper
                print("[INFO] calibrado HSV mano:", lower, upper)
            else:
                print("[WARN] ROI de mano demasiado pequeño")
        else:
            print("[WARN] dibuja primero un ROI en la ventana 'Mano'")

    # guardar gesto
    elif key == ord('g'):
        if feat_vec is not None:
            save_gesture_example(feat_vec, state["current_label"])
            if RECOGNIZE_MODE:
                state["gallery"].append((feat_vec, state["current_label"]))
            print(f"[INFO] guardado gesto con etiqueta {state['current_label']}")
        else:
            print("[WARN] no hay gesto válido para guardar")

    # añadir gesto reconocido a secuencia
    elif key == ord('a'):
        state["acciones"] = ui.append_action(state["acciones"], state["stable_label"])

    # guardar secuencia de gestos
    elif key == ord('p'):
        save_sequence_json(state["acciones"])
        print("[INFO] secuencia guardada:", state["acciones"])
        state["acciones"].clear()

    # cambiar etiqueta actual
    elif key in (ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
        mapping = {
            ord('0'): "0dedos",
            ord('1'): "1dedo",
            ord('2'): "2dedos",
            ord('3'): "3dedos",
            ord('4'): "4dedos",
            ord('5'): "5dedos",
        }
        state["current_label"] = mapping[key]

    return state
