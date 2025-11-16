# board_processing.py
import cv2
import numpy as np
import board_tracker
import object_tracker
import board_ui


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
            ship_two_mask_show, ship_one_mask_show, layout_info = process_single_board(
                vis_all, frame, quad, slot, warp_size
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


def process_single_board(vis_img, frame_bgr, quad, slot, warp_size=500):
    """
    Procesa SOLO un tablero:
    - aplanado
    - detección de barcos por color
    - pintado de las celdas ocupadas
    """

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
    warp_hsv = cv2.cvtColor(warp_img, cv2.COLOR_BGR2HSV)

    ship_two_mask, ship_two_cells, ship_two_centers = detect_ship_cells(
        warp_hsv,
        object_tracker.current_ship_two_lower,
        object_tracker.current_ship_two_upper,
        min_area=150,
        max_cells_per_contour=2,
    )

    ship_one_mask, ship_one_cells, ship_one_centers = detect_ship_cells(
        warp_hsv,
        object_tracker.current_ship_one_lower,
        object_tracker.current_ship_one_upper,
        min_area=50,
        max_cells_per_contour=1,
    )

    slot["ship_two_cells"] = ship_two_cells
    slot["ship_one_cells"] = ship_one_cells

    draw_cells_on_warp(warp_img, ship_two_cells, (0, 0, 255))
    draw_cells_on_warp(warp_img, ship_one_cells, (0, 255, 255))
    draw_centers_on_warp(warp_img, ship_two_centers, (0, 0, 255))
    draw_centers_on_warp(warp_img, ship_one_centers, (0, 255, 255))
    draw_cells_on_board(vis_img, H_inv, ship_two_cells, (0, 0, 255), warp_size)
    draw_cells_on_board(vis_img, H_inv, ship_one_cells, (0, 255, 255), warp_size)

    cv2.imshow(f"{slot['name']} aplanado", warp_img)

    layout_info = {
        "name": slot["name"],
        "ship_two_cells": ship_two_cells,
        "ship_one_cells": ship_one_cells,
        "board_size": board_tracker.BOARD_SQUARES,
    }

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


def detect_ship_cells(
    warp_hsv,
    lower,
    upper,
    min_area=60,
    max_cells_per_contour=None,
):
    mask = cv2.inRange(warp_hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    n = board_tracker.BOARD_SQUARES
    h, w = mask.shape[:2]
    cell_w = w / n if n else w
    cell_h = h / n if n else h

    cells = []
    centers = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

        bbox_cells = _cells_from_bounding_box(c, cell_w, cell_h, n)
        if not bbox_cells:
            row = _clip_index(int(cy / cell_h) if n else 0, n)
            col = _clip_index(int(cx / cell_w) if n else 0, n)
            bbox_cells = [(row, col)]

        if max_cells_per_contour is not None:
            bbox_cells = _limit_cells_near_center(bbox_cells, (cx, cy), cell_w, cell_h, max_cells_per_contour)

        cells.extend(bbox_cells)

    cells = sorted(set(cells))
    return mask, cells, centers


def _cells_from_bounding_box(contour, cell_w, cell_h, n):
    if n <= 0:
        return []
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return []

    col_start = _clip_index(int(np.floor(x / cell_w)), n)
    col_end = _clip_index(int(np.floor((x + w - 1) / cell_w)), n)
    row_start = _clip_index(int(np.floor(y / cell_h)), n)
    row_end = _clip_index(int(np.floor((y + h - 1) / cell_h)), n)

    cells = []
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            cells.append((row, col))
    return cells


def _limit_cells_near_center(cells, center, cell_w, cell_h, limit):
    if len(cells) <= limit:
        return cells

    cx, cy = center

    def cell_distance(cell):
        row, col = cell
        cell_cx = (col + 0.5) * cell_w
        cell_cy = (row + 0.5) * cell_h
        return (cell_cx - cx) ** 2 + (cell_cy - cy) ** 2

    sorted_cells = sorted(cells, key=cell_distance)
    return sorted_cells[:limit]


def _clip_index(idx, n):
    if n <= 0:
        return 0
    return max(0, min(n - 1, idx))


def draw_cells_on_warp(warp_img, cells, color):
    n = board_tracker.BOARD_SQUARES
    h, w = warp_img.shape[:2]
    cell_w = w / n if n else w
    cell_h = h / n if n else h
    for row, col in cells:
        x0 = int(col * cell_w)
        x1 = int((col + 1) * cell_w)
        y0 = int(row * cell_h)
        y1 = int((row + 1) * cell_h)
        cv2.rectangle(warp_img, (x0, y0), (x1, y1), color, 2)


def draw_cells_on_board(vis_img, H_inv, cells, color, warp_size):
    if not cells:
        return
    n = board_tracker.BOARD_SQUARES
    cell_w = warp_size / n if n else warp_size
    for row, col in cells:
        x0 = col * cell_w
        x1 = (col + 1) * cell_w
        y0 = row * cell_w
        y1 = (row + 1) * cell_w
        quad = np.array(
            [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ],
            dtype=np.float32,
        )
        pts = cv2.perspectiveTransform(quad[None, :, :], H_inv)[0]
        cv2.polylines(vis_img, [pts.astype(int)], True, color, 2)


def draw_centers_on_warp(warp_img, centers, color):
    for cx, cy in centers:
        cv2.circle(warp_img, (int(cx), int(cy)), 6, color, 2)
