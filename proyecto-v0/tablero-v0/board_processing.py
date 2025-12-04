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


def procesar_todos_los_tableros(
    frame_bgr,
    lista_estados_tablero,
    matriz_camara=None,
    coef_distorsion=None,
    max_tableros=2,
    tamano_warp=500,
    imprimir_detecciones=True,
):
    """
    Detecta varios tableros, los asigna a los slots existentes (T1, T2),
    procesa cada uno y devuelve todo para mostrar.
    """
    vis_all, tableros_detectados, mascara_tablero = board_tracker.detectar_tableros_multiples(
        frame_bgr,
        matriz_camara=matriz_camara,
        coeficientes_distorsion=coef_distorsion,
        max_tableros=max_tableros,
    )

    # dibujar ROI y HUD
    board_ui.dibujar_roi_tablero(vis_all)
    board_ui.dibujar_hud_tablero(vis_all)

    # asignar detecciones a slots por cercanía
    asignaciones = _asignar_detecciones_a_slots(tableros_detectados, lista_estados_tablero)

    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    puntos_municion, mascara_municion = object_tracker.detectar_puntos_coloreados_global(
        frame_hsv,
        object_tracker.rango_inferior_municion,
        object_tracker.rango_superior_municion,
        max_objetos=12,
        area_minima=30,
    )
    for (cx, cy) in puntos_municion:
        cv2.circle(vis_all, (cx, cy), 6, (255, 0, 255), -1)

    mascara_barco_doble = None
    mascara_barco_simple = None
    distribuciones = []

    for indice_slot, slot in enumerate(lista_estados_tablero):
        indice_det = asignaciones.get(indice_slot, None)
        if indice_det is not None:
            info_tablero = tableros_detectados[indice_det]
            quad = info_tablero["quad"]
            slot["last_quad"] = quad
            slot["miss"] = 0
            if info_tablero.get("ratio") is not None:
                slot["cm_per_pix"] = info_tablero["ratio"]
            mascara_barco_doble, mascara_barco_simple, info_distribucion = procesar_tablero(
                vis_all,
                frame_bgr,
                quad,
                slot,
                tamano_warp,
                slot.get("cm_per_pix"),
                imprimir_detecciones=imprimir_detecciones,
            )
            if info_distribucion is not None:
                distribuciones.append(info_distribucion)
        else:
            degradar_estado_tablero(slot, vis_all)

    return (
        vis_all,
        mascara_tablero,
        mascara_barco_doble,
        mascara_barco_simple,
        mascara_municion,
        distribuciones,
    )


def _asignar_detecciones_a_slots(tableros_detectados, lista_estados_tablero):
    """
    Empareja detecciones de tableros con los slots (T1, T2) por proximidad.
    Así no cambian de nombre cuando el contorno baila.
    """
    asignaciones = {}
    if not tableros_detectados:
        return asignaciones

    # centros de detección
    centros_detectados = []
    for tablero in tableros_detectados:
        quad = tablero["quad"]
        cx = np.mean(quad[:, 0])
        cy = np.mean(quad[:, 1])
        centros_detectados.append((cx, cy))

    used = set()
    for indice_slot, slot in enumerate(lista_estados_tablero):
        mejor_det = None
        mejor_distancia = 1e9

        if slot["last_quad"] is not None:
            sq = slot["last_quad"]
            sx = np.mean(sq[:, 0])
            sy = np.mean(sq[:, 1])
            centro_slot = (sx, sy)
        else:
            centro_slot = None

        for indice_det, (dx, dy) in enumerate(centros_detectados):
            if indice_det in used:
                continue
            if centro_slot is None:
                mejor_det = indice_det
                break
            distancia = ((dx - centro_slot[0]) ** 2 + (dy - centro_slot[1]) ** 2) ** 0.5
            if distancia < mejor_distancia:
                mejor_distancia = distancia
                mejor_det = indice_det

        if mejor_det is not None:
            asignaciones[indice_slot] = mejor_det
            used.add(mejor_det)

    return asignaciones


def procesar_tablero(
    vis_img,
    frame_bgr,
    quad,
    slot,
    tamano_warp=500,
    cm_por_pixel=None,
    imprimir_detecciones=True,
):
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
            [tamano_warp - 1, 0],
            [tamano_warp - 1, tamano_warp - 1],
            [0, tamano_warp - 1],
        ],
        dtype=np.float32,
    )
    H_warp = cv2.getPerspectiveTransform(src, dst)
    H_inv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(frame_bgr, H_warp, (tamano_warp, tamano_warp))

    puntos_barco_doble, mascara_barco_doble = object_tracker.detectar_puntos_coloreados_en_tablero(
        hsv,
        quad,
        object_tracker.rango_inferior_barco_doble,
        object_tracker.rango_superior_barco_doble,
        max_objetos=4,
        area_minima=40,
    )

    puntos_barco_simple, mascara_barco_simple = object_tracker.detectar_puntos_coloreados_en_tablero(
        hsv,
        quad,
        object_tracker.rango_inferior_barco_simple,
        object_tracker.rango_superior_barco_simple,
        max_objetos=6,
        area_minima=40,
    )

    _dibujar_puntos(vis_img, puntos_barco_doble, (0, 0, 255))
    _dibujar_puntos(vis_img, puntos_barco_simple, (0, 255, 255))
    _dibujar_puntos_en_warp(warp_img, puntos_barco_doble, H_warp, (0, 0, 255))
    _dibujar_puntos_en_warp(warp_img, puntos_barco_simple, H_warp, (0, 255, 255))

    celdas_barco_doble, etiquetas_barco_doble = _mapear_puntos_a_celdas(
        puntos_barco_doble, H_warp, tamano_warp
    )
    celdas_barco_simple, etiquetas_barco_simple = _mapear_puntos_a_celdas(
        puntos_barco_simple, H_warp, tamano_warp
    )

    slot["ship_two_cells"] = sorted(set(celdas_barco_doble))
    slot["ship_one_cells"] = sorted(set(celdas_barco_simple))
    slot["ship_two_points"] = list(puntos_barco_doble)
    slot["ship_one_points"] = list(puntos_barco_simple)

    entradas_visualizacion = []
    entradas_impresion = []
    for idx, (pt, label) in enumerate(zip(puntos_barco_doble, etiquetas_barco_doble), 1):
        entradas_visualizacion.append((f"B2-{idx}", label))
        entradas_impresion.append((f"B2-{idx}", label, _formatear_desplazamiento_origen(pt, cm_por_pixel)))
    for idx, (pt, label) in enumerate(zip(puntos_barco_simple, etiquetas_barco_simple), 1):
        entradas_visualizacion.append((f"B1-{idx}", label))
        entradas_impresion.append((f"B1-{idx}", label, _formatear_desplazamiento_origen(pt, cm_por_pixel)))

    _anotar_detecciones(vis_img, warp_img, slot["name"], entradas_visualizacion)

    info_distribucion = {
        "name": slot["name"],
        "ship_two_cells": slot["ship_two_cells"],
        "ship_one_cells": slot["ship_one_cells"],
        "board_size": board_tracker.CASILLAS_TABLERO,
    }

    if imprimir_detecciones and entradas_visualizacion:
        key = (slot["name"], tuple(entradas_impresion))
        if last_display_entries.get(slot["name"]) != key:
            for tag, label, offset_txt in entradas_impresion:
                if offset_txt:
                    print(f"[{slot['name']}] {tag} -> {label} | {offset_txt}")
                else:
                    print(f"[{slot['name']}] {tag} -> {label}")
            last_display_entries[slot["name"]] = key

    cv2.imshow(f"{slot['name']} aplanado", warp_img)

    return mascara_barco_doble, mascara_barco_simple, info_distribucion


def degradar_estado_tablero(slot, vis_img):
    if slot["last_quad"] is not None and slot["miss"] <= 10:
        dibujar_cuadricula(vis_img, slot["last_quad"])
        slot["miss"] += 1
    else:
        slot["miss"] += 1
        slot["ship_two_cells"] = []
        slot["ship_one_cells"] = []
        slot["ship_two_points"] = []
        slot["ship_one_points"] = []


def dibujar_cuadricula(img, quad, color=(0, 255, 255)):
    if quad is None:
        return
    q = np.array(quad, dtype=np.int32)
    cv2.polylines(img, [q], True, color, 2)


def _mapear_puntos_a_celdas(puntos, H_warp, tamano_warp):
    if not puntos:
        return [], []

    pts = np.array(puntos, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H_warp).reshape(-1, 2)
    n = board_tracker.CASILLAS_TABLERO
    if n <= 0:
        return [], []
    tamano_celda = tamano_warp / n

    celdas = []
    etiquetas = []

    # Anclamos el origen en la esquina superior izquierda (A1) y
    # avanzamos letras hacia la derecha y números hacia abajo.
    def _celda_desde_eje(coord):
        return _acotar_indice_celda(int(np.floor(coord / tamano_celda)), n)

    for wx, wy in warped:
        col = _celda_desde_eje(wx)
        row = _celda_desde_eje(wy)
        celdas.append((row, col))
        etiquetas.append(_formatear_etiqueta_celda(row, col))
    return celdas, etiquetas


def _dibujar_puntos(img, puntos, color):
    for (cx, cy) in puntos:
        cv2.circle(img, (int(cx), int(cy)), 6, color, -1)


def _dibujar_puntos_en_warp(warp_img, puntos, H_warp, color):
    if not puntos:
        return
    pts = np.array(puntos, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H_warp).reshape(-1, 2)
    for wx, wy in warped:
        cv2.circle(warp_img, (int(wx), int(wy)), 6, color, 2)


def _anotar_detecciones(vis_img, warp_img, nombre_slot, entradas):
    if not entradas:
        return

    desplazamiento_y = 120 if nombre_slot == "T1" else 220
    for tag, label in entradas:
        texto = f"{nombre_slot}-{tag}: {label}"
        cv2.putText(
            vis_img,
            texto,
            (10, desplazamiento_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        desplazamiento_y += 18

    for idx, (tag, label) in enumerate(entradas):
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


def _formatear_desplazamiento_origen(punto, cm_por_pixel=None):
    if board_state.ORIGEN_GLOBAL is None:
        return None

    ox, oy = board_state.ORIGEN_GLOBAL
    px_dx = punto[0] - ox
    px_dy = punto[1] - oy
    px_dist = math.hypot(px_dx, px_dy)

    if cm_por_pixel:
        cm_dx = px_dx * cm_por_pixel
        cm_dy = px_dy * cm_por_pixel
        cm_dist = px_dist * cm_por_pixel
        return f"offset ArUco dx={cm_dx:.1f}cm dy={cm_dy:.1f}cm dist={cm_dist:.1f}cm"

    return f"offset ArUco dx={px_dx:.1f}px dy={px_dy:.1f}px dist={px_dist:.1f}px"


def limpiar_cache_visualizacion():
    last_display_entries.clear()


def _formatear_etiqueta_celda(row, col):
    return f"{chr(ord('A') + col)}{row + 1}"


def _acotar_indice_celda(idx, n):
    return max(0, min(n - 1, idx))
