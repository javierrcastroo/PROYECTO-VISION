# board_processing.py
import cv2
import math
import numpy as np

import board_state
import board_tracker
import object_tracker
import board_ui

# Recuerda el último listado de detecciones por tablero para evitar spam en consola
last_display_entries = {}


def process_all_boards(frame, boards_state_list, cam_mtx=None, dist=None, max_boards=2, warp_size=500):
    """
    Detecta varios tableros, los asigna a los slots existentes (T1, T2),
    procesa cada uno y devuelve todo para mostrar.
    """
    vis_all, boards_found, mask_board = board_tracker.detect_multiple_boards(
        frame,
        camera_matrix=cam_mtx,
        dist_coeffs=dist,
        max_boards=max_boards,
    )

    # dibujar ROI y HUD
    board_ui.draw_board_roi(vis_all)
    board_ui.draw_board_hud(vis_all)

    # asignar detecciones a slots por cercanía
    assignments = _assign_detections_to_slots(boards_found, boards_state_list)

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ammo_pts, ammo_mask_show = object_tracker.detect_colored_points_global(
        frame_hsv,
        object_tracker.current_ammo_lower,
        object_tracker.current_ammo_upper,
        max_objs=12,
        min_area=30,
    )
    for (cx, cy) in ammo_pts:
        cv2.circle(vis_all, (cx, cy), 6, (255, 0, 255), -1)

    ship_two_mask_show = None
    ship_one_mask_show = None
    layouts = []

    for slot_idx, slot in enumerate(boards_state_list):
        det_idx = assignments.get(slot_idx, None)
        if det_idx is not None:
            binfo = boards_found[det_idx]
            quad = binfo["quad"]
            slot["last_quad"] = quad
            slot["miss"] = 0
            if binfo.get("ratio") is not None:
                slot["cm_per_pix"] = binfo["ratio"]
            ship_two_mask_show, ship_one_mask_show, layout_info = process_single_board(
                vis_all, frame, quad, slot, warp_size, slot.get("cm_per_pix")
            )
            if layout_info is not None:
                layouts.append(layout_info)
        else:
            fallback_or_decay(slot, vis_all)

    return (
        vis_all,
        mask_board,
        ship_two_mask_show,
        ship_one_mask_show,
        ammo_mask_show,
        layouts,
    )


def _assign_detections_to_slots(boards_found, boards_state_list):
    """
    Empareja detecciones de tableros con los slots (T1, T2) por proximidad.
    Así no cambian de nombre cuando el contorno baila.
    """
    assignments = {}
    if not boards_found:
        return assignments

    # centros de detección
    det_centers = []
    for b in boards_found:
        quad = b["quad"]
        cx = np.mean(quad[:, 0])
        cy = np.mean(quad[:, 1])
        det_centers.append((cx, cy))

    used = set()
    for slot_idx, slot in enumerate(boards_state_list):
        best_det = None
        best_dist = 1e9

        if slot["last_quad"] is not None:
            sq = slot["last_quad"]
            sx = np.mean(sq[:, 0])
            sy = np.mean(sq[:, 1])
            slot_center = (sx, sy)
        else:
            slot_center = None

        for det_idx, (dx, dy) in enumerate(det_centers):
            if det_idx in used:
                continue
            if slot_center is None:
                best_det = det_idx
                break
            dist = ((dx - slot_center[0]) ** 2 + (dy - slot_center[1]) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_det = det_idx

        if best_det is not None:
            assignments[slot_idx] = best_det
            used.add(best_det)

    return assignments


def process_single_board(vis_img, frame_bgr, quad, slot, warp_size=500, cm_per_pix=None):
    """
    Procesa un tablero individual detectando centros de barcos de dos y una casilla
    con el mismo pipeline basado en blobs que teníamos antes: calibras con un ROI,
    buscamos contornos del color elegido, calculamos su centroide y lo traducimos
    a una casilla (A1, B2, ...).
    """

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    src = np.array(quad, dtype=np.float32)
    dst = np.array(
        [
            [0, 0],
            [warp_size - 1, 0],
            [warp_size - 1, warp_size - 1],
            [0, warp_size - 1],
        ],
        dtype=np.float32,
    )
    H_warp = cv2.getPerspectiveTransform(src, dst)
    H_inv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(frame_bgr, H_warp, (warp_size, warp_size))

    ship_two_pts, ship_two_mask = object_tracker.detect_colored_points_in_board(
        hsv,
        quad,
        object_tracker.current_ship_two_lower,
        object_tracker.current_ship_two_upper,
        max_objs=4,
        min_area=40,
    )

    ship_one_pts, ship_one_mask = object_tracker.detect_colored_points_in_board(
        hsv,
        quad,
        object_tracker.current_ship_one_lower,
        object_tracker.current_ship_one_upper,
        max_objs=6,
        min_area=40,
    )

    _draw_points(vis_img, ship_two_pts, (0, 0, 255))
    _draw_points(vis_img, ship_one_pts, (0, 255, 255))
    _draw_points_on_warp(warp_img, ship_two_pts, H_warp, (0, 0, 255))
    _draw_points_on_warp(warp_img, ship_one_pts, H_warp, (0, 255, 255))

    ship_two_cells_raw, ship_two_labels = _map_points_to_cells(
        ship_two_pts, H_warp, warp_size
    )
    ship_one_cells_raw, ship_one_labels = _map_points_to_cells(
        ship_one_pts, H_warp, warp_size
    )

    slot["ship_two_cells"] = sorted(set(ship_two_cells_raw))
    slot["ship_one_cells"] = sorted(set(ship_one_cells_raw))

    display_entries = []
    printable_entries = []
    for idx, (pt, label) in enumerate(zip(ship_two_pts, ship_two_labels), 1):
        display_entries.append((f"B2-{idx}", label))
        printable_entries.append((f"B2-{idx}", label, _format_origin_offset(pt, cm_per_pix)))
    for idx, (pt, label) in enumerate(zip(ship_one_pts, ship_one_labels), 1):
        display_entries.append((f"B1-{idx}", label))
        printable_entries.append((f"B1-{idx}", label, _format_origin_offset(pt, cm_per_pix)))

    _annotate_detections(vis_img, warp_img, slot["name"], display_entries)

    layout_info = {
        "name": slot["name"],
        "ship_two_cells": slot["ship_two_cells"],
        "ship_one_cells": slot["ship_one_cells"],
        "board_size": board_tracker.BOARD_SQUARES,
    }

    if display_entries:
        key = (slot["name"], tuple(printable_entries))
        if last_display_entries.get(slot["name"]) != key:
            for tag, label, offset_txt in printable_entries:
                if offset_txt:
                    print(f"[{slot['name']}] {tag} -> {label} | {offset_txt}")
                else:
                    print(f"[{slot['name']}] {tag} -> {label}")
            last_display_entries[slot["name"]] = key

    cv2.imshow(f"{slot['name']} aplanado", warp_img)

    return ship_two_mask, ship_one_mask, layout_info


def fallback_or_decay(slot, vis_img):
    if slot["last_quad"] is not None and slot["miss"] <= 10:
        draw_quad(vis_img, slot["last_quad"])
        slot["miss"] += 1
    else:
        slot["miss"] += 1
        slot["ship_two_cells"] = []
        slot["ship_one_cells"] = []


def draw_quad(img, quad, color=(0, 255, 255)):
    if quad is None:
        return
    q = np.array(quad, dtype=np.int32)
    cv2.polylines(img, [q], True, color, 2)


def _map_points_to_cells(points, H_warp, warp_size):
    if not points:
        return [], []

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H_warp).reshape(-1, 2)
    n = board_tracker.BOARD_SQUARES
    if n <= 0:
        return [], []
    cell_size = warp_size / n

    cells = []
    labels = []
    for wx, wy in warped:
        col = _clip_cell_index(int(np.floor(wx / cell_size)), n)
        row = _clip_cell_index(int(np.floor(wy / cell_size)), n)
        cells.append((row, col))
        labels.append(_format_cell_label(row, col))
    return cells, labels


def _draw_points(img, points, color):
    for (cx, cy) in points:
        cv2.circle(img, (int(cx), int(cy)), 6, color, -1)


def _draw_points_on_warp(warp_img, points, H_warp, color):
    if not points:
        return
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H_warp).reshape(-1, 2)
    for wx, wy in warped:
        cv2.circle(warp_img, (int(wx), int(wy)), 6, color, 2)


def _annotate_detections(vis_img, warp_img, slot_name, entries):
    if not entries:
        return

    y_offset = 120 if slot_name == "T1" else 220
    for tag, label in entries:
        text = f"{slot_name}-{tag}: {label}"
        cv2.putText(
            vis_img,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        y_offset += 18

    for idx, (tag, label) in enumerate(entries):
        base_y = 25 + idx * 22
        cv2.rectangle(warp_img, (10, base_y - 15), (260, base_y + 5), (0, 0, 0), -1)
        cv2.putText(
            warp_img,
            f"{tag}: {label}",
            (15, base_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _format_origin_offset(point, cm_per_pix=None):
    if board_state.GLOBAL_ORIGIN is None:
        return None

    ox, oy = board_state.GLOBAL_ORIGIN
    px_dx = point[0] - ox
    px_dy = point[1] - oy
    px_dist = math.hypot(px_dx, px_dy)

    if cm_per_pix:
        cm_dx = px_dx * cm_per_pix
        cm_dy = px_dy * cm_per_pix
        cm_dist = px_dist * cm_per_pix
        return f"offset ArUco dx={cm_dx:.1f}cm dy={cm_dy:.1f}cm dist={cm_dist:.1f}cm"

    return f"offset ArUco dx={px_dx:.1f}px dy={px_dy:.1f}px dist={px_dist:.1f}px"


def clear_display_cache():
    last_display_entries.clear()


def _format_cell_label(row, col):
    return f"{chr(ord('A') + col)}{row + 1}"


def _clip_cell_index(idx, n):
    return max(0, min(n - 1, idx))
