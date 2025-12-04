# hand_main.py
import cv2
import json
import os
import time
import numpy as np

from hand_config import (
    PREVIEW_W,
    PREVIEW_H,
    RECOGNIZE_MODE,
    CONFIDENCE_THRESHOLD,
    USE_UNDISTORT_HAND,
    HAND_CAMERA_PARAMS_PATH,
    ATTACKS_DIR,
)

import ui
from segmentation import calibrar_desde_roi, segmentar_mascara_mano
from features import calcular_vector_caracteristicas
from classifier import predecir_knn
from storage import (
    guardar_ejemplo_gesto,
    cargar_galeria_gestos,
    guardar_secuencia_json,
    guardar_peticion_reinicio,
)
from collections import deque

VENTANA_GESTOS_FRAMES = 150
LONGITUD_MAXIMA_SECUENCIA = 2
GESTOS_ACTIVACION = {"5dedos"}
GESTO_CONFIRMAR = "ok"
GESTO_RECHAZAR = "nook"
GESTO_IMPRIMIR = "cool"
GESTO_REINICIO = "demond"
GESTOS_CONTROL = GESTOS_ACTIVACION | {GESTO_CONFIRMAR, GESTO_RECHAZAR, GESTO_IMPRIMIR}
TABLERO_OBJETIVO = os.environ.get("BATTLESHIP_TARGET", "1")
ARCHIVO_RETROALIMENTACION = os.path.join(ATTACKS_DIR, "last_result.json")

MAPA_COORDENADAS = {
    "0dedos": 0,
    "1dedo": 1,
    "2dedos": 2,
    "3dedos": 3,
    "4dedos": 4,
}


def secuencia_a_coordenada(secuencia):
    if len(secuencia) != 2:
        return None
    etiqueta_columna, etiqueta_fila = secuencia
    if etiqueta_columna not in MAPA_COORDENADAS or etiqueta_fila not in MAPA_COORDENADAS:
        return None
    columna = MAPA_COORDENADAS[etiqueta_columna]
    fila = MAPA_COORDENADAS[etiqueta_fila]
    return fila, columna


def voto_mayoritario(etiquetas):
    if not etiquetas:
        return None
    return max(set(etiquetas), key=etiquetas.count)


class VentanaGestos:
    def __init__(self, tamano=VENTANA_GESTOS_FRAMES):
        self.tamano = tamano
        self.etiquetas = []

    def reiniciar(self):
        self.etiquetas = []

    def agregar(self, etiqueta):
        etiqueta = etiqueta if etiqueta is not None else "????"
        self.etiquetas.append(etiqueta)
        if len(self.etiquetas) >= self.tamano:
            ganadora = voto_mayoritario(self.etiquetas)
            self.reiniciar()
            return ganadora
        return None

    def progreso(self):
        if self.tamano == 0:
            return 0.0
        return min(1.0, len(self.etiquetas) / float(self.tamano))


def cargar_ultimo_resultado(archivo_feedback, ultima_marca):
    if not os.path.exists(archivo_feedback):
        return ultima_marca, None, None

    try:
        marca_tiempo = os.path.getmtime(archivo_feedback)
    except OSError:
        return ultima_marca, None, None

    if marca_tiempo <= ultima_marca:
        return ultima_marca, None, None

    try:
        with open(archivo_feedback, "r", encoding="utf-8") as archivo:
            payload = json.load(archivo)
    except (OSError, json.JSONDecodeError):
        return ultima_marca, None, None

    mensajes = payload.get("messages")
    if not mensajes:
        respaldo = payload.get("status") or "Resultado recibido"
        mensajes = [respaldo]

    return marca_tiempo, mensajes, payload


def main():
    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        raise RuntimeError("No se pudo abrir la camara 0 (mano)")

    matriz_camara_mano = distorsion_mano = None
    if USE_UNDISTORT_HAND and os.path.exists(HAND_CAMERA_PARAMS_PATH):
        datos = np.load(HAND_CAMERA_PARAMS_PATH)
        matriz_camara_mano = datos["camera_matrix"]
        distorsion_mano = datos["dist_coeffs"]
        print("[INFO] Undistort activado para la mano")

    # estado
    piel_inferior = piel_superior = None
    galeria = cargar_galeria_gestos() if RECOGNIZE_MODE else []
    etiqueta_actual = "2dedos"
    acciones = []
    predicciones_recientes = deque(maxlen=7)
    estado_captura = "ESPERA"
    candidato_pendiente = None
    ventana_gestos = VentanaGestos()
    lineas_estado = ["En espera: haz '5dedos' para activar el registro."]
    tablero_objetivo_actual = "T2"
    contadores_ataques = {"T1": 0, "T2": 0}
    lineas_retroalimentacion = []
    ultima_modificacion_feedback = 0.0
    partida_finalizada = False

    def establecer_estado(nuevo_estado, lineas):
        nonlocal estado_captura, lineas_estado
        estado_captura = nuevo_estado
        lineas_estado = lineas
        ventana_gestos.reiniciar()

    def actualizar_lineas_estado(lineas):
        nonlocal lineas_estado
        lineas_estado = lineas

    cv2.namedWindow("Mano")
    cv2.setMouseCallback("Mano", ui.callback_raton)

    while True:
        ok, fotograma = captura.read()
        if not ok:
            break

        # undistort
        if matriz_camara_mano is not None:
            fotograma = cv2.undistort(fotograma, matriz_camara_mano, distorsion_mano)

        # espejo + resize
        fotograma = cv2.flip(fotograma, 1)
        fotograma = cv2.resize(fotograma, (PREVIEW_W, PREVIEW_H))
        visualizacion = fotograma.copy()
        hsv = cv2.cvtColor(fotograma, cv2.COLOR_BGR2HSV)

        # ROI
        ui.dibujar_rectangulo_roi(visualizacion)

        # segmentar mano con el HSV calibrado
        mascara = segmentar_mascara_mano(hsv, piel_inferior, piel_superior)
        ui.dibujar_caja_mano(visualizacion, mascara)
        piel_sola = cv2.bitwise_and(fotograma, fotograma, mask=mascara)

        # features
        vector_caracteristicas = calcular_vector_caracteristicas(mascara)

        # reconocimiento
        mejor_distancia = None
        etiqueta_por_frame = None
        if vector_caracteristicas is not None and RECOGNIZE_MODE:
            etiqueta_bruta, mejor_distancia = predecir_knn(vector_caracteristicas, galeria, k=5)
            if etiqueta_bruta is not None and mejor_distancia is not None:
                etiqueta_por_frame = etiqueta_bruta if mejor_distancia <= CONFIDENCE_THRESHOLD else "????"

        if etiqueta_por_frame is not None:
            predicciones_recientes.append(etiqueta_por_frame)
        etiqueta_estable = voto_mayoritario(list(predicciones_recientes))

        # HUD
        ui.dibujar_hud(
            visualizacion,
            piel_inferior,
            piel_superior,
            etiqueta_actual,
        )
        ui.dibujar_prediccion(visualizacion, etiqueta_estable, mejor_distancia if mejor_distancia else 0.0)

        ultima_modificacion_feedback, nueva_retroalimentacion, metadata_feedback = cargar_ultimo_resultado(
            ARCHIVO_RETROALIMENTACION, ultima_modificacion_feedback
        )
        if nueva_retroalimentacion:
            lineas_retroalimentacion = nueva_retroalimentacion
            for linea in lineas_retroalimentacion:
                print(f"[RESULTADO] {linea}")

            siguiente_objetivo = None
            if metadata_feedback:
                siguiente_objetivo = metadata_feedback.get("next_defender") or metadata_feedback.get("defender")
                ganador = metadata_feedback.get("winner")
                estado_meta = metadata_feedback.get("status")
                partida_finalizada = bool(ganador) or estado_meta == "finished"
                if estado_meta == "reset":
                    partida_finalizada = False
                    contadores_ataques = {"T1": 0, "T2": 0}
            if siguiente_objetivo and siguiente_objetivo != tablero_objetivo_actual:
                tablero_objetivo_actual = siguiente_objetivo
                actualizar_lineas_estado(
                    [
                        f"Objetivo segun tablero: {tablero_objetivo_actual}",
                        "En espera: haz '5dedos' para activar el registro.",
                    ]
                )
        else:
            if metadata_feedback and metadata_feedback.get("status") == "turn":
                partida_finalizada = False

        lineas_a_mostrar = lineas_estado + lineas_retroalimentacion
        ui.dibujar_estado_secuencia(
            visualizacion,
            acciones,
            estado_captura,
            candidato_pendiente,
            lineas_a_mostrar,
            ventana_gestos.progreso(),
        )

        # mostrar
        cv2.imshow("Mano", visualizacion)
        cv2.imshow("Mascara mano", mascara)
        cv2.imshow("Solo piel mano", piel_sola)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (27, ord('q')):
            break

        # -------- flujo controlado por gestos --------
        etiqueta_resuelta = ventana_gestos.agregar(etiqueta_estable)

        if etiqueta_resuelta is not None:
            if partida_finalizada and etiqueta_resuelta == GESTO_REINICIO:
                guardar_peticion_reinicio()
                acciones.clear()
                candidato_pendiente = None
                contadores_ataques = {"T1": 0, "T2": 0}
                partida_finalizada = False
                establecer_estado(
                    "ESPERA",
                    [
                        "Reinicio solicitado. Espera a que el tablero prepare nueva partida.",
                        "En espera: haz '5dedos' para activar el registro.",
                    ],
                )
                continue

            if estado_captura == "ESPERA":
                if etiqueta_resuelta in GESTOS_ACTIVACION:
                    establecer_estado("CAPTURA", ["Sistema activo: muestra el primer gesto."])
                else:
                    actualizar_lineas_estado(["Sigue en espera, haz '5dedos' para comenzar."])

            elif estado_captura == "CAPTURA":
                if etiqueta_resuelta == "????" or etiqueta_resuelta in GESTOS_CONTROL:
                    actualizar_lineas_estado(["Gesto no valido, repitelo."])
                else:
                    candidato_pendiente = etiqueta_resuelta
                    establecer_estado(
                        "CONFIRMACION",
                        [
                            f"Tu gesto es '{candidato_pendiente}'?",
                            "Confirma con 'ok' o repite con 'nook'.",
                        ],
                    )

            elif estado_captura == "CONFIRMACION":
                if etiqueta_resuelta == GESTO_CONFIRMAR and candidato_pendiente:
                    acciones.append(candidato_pendiente)
                    print(f"[INFO] Aniadido gesto confirmado: {candidato_pendiente}")
                    candidato_pendiente = None
                    if len(acciones) >= LONGITUD_MAXIMA_SECUENCIA:
                        establecer_estado(
                            "IMPRESION",
                            ["Secuencia completa, haz 'cool' para imprimirla."],
                        )
                    else:
                        establecer_estado("CAPTURA", ["Gesto guardado. Muestra el siguiente gesto."])
                elif etiqueta_resuelta == GESTO_RECHAZAR:
                    print("[INFO] Gesto rechazado, repite el anterior.")
                    candidato_pendiente = None
                    establecer_estado("CAPTURA", ["Repite el gesto a registrar."])
                else:
                    actualizar_lineas_estado(["Se esperaba 'ok' o 'nook'."])

            elif estado_captura == "IMPRESION":
                if etiqueta_resuelta == GESTO_IMPRIMIR and len(acciones) == LONGITUD_MAXIMA_SECUENCIA:
                    coord = secuencia_a_coordenada(acciones)
                    if coord is None:
                        actualizar_lineas_estado(
                            [
                                "Secuencia invalida para coordenada (usa 0-4 dedos).",
                                "Repite los dos gestos de columna y fila.",
                            ]
                        )
                    else:
                        fila, columna = coord
                        contadores_ataques[tablero_objetivo_actual] += 1
                        disparo = contadores_ataques[tablero_objetivo_actual]
                        print("[INFO] Secuencia final:", acciones)
                        guardar_secuencia_json(
                            acciones,
                            nombre_objetivo=tablero_objetivo_actual,
                            numero_disparo=disparo,
                            fila=fila,
                            columna=columna,
                        )
                        acciones.clear()
                        candidato_pendiente = None
                        establecer_estado(
                            "ESPERA",
                            [
                                "En espera: haz '5dedos' para activar un nuevo registro.",
                                f"Objetivo actual: {tablero_objetivo_actual}",
                            ],
                        )
                else:
                    actualizar_lineas_estado(["Secuencia lista. Usa 'cool' para imprimirla."])

        # -------- teclas de mano --------
        if tecla == ord('c'):

            if ui.roi_definido:
                x0, x1 = sorted([ui.x_inicio, ui.x_fin])
                y0, y1 = sorted([ui.y_inicio, ui.y_fin])
                if (x1 - x0) > 5 and (y1 - y0) > 5:
                    roi_hsv = hsv[y0:y1, x0:x1]
                    piel_inferior, piel_superior = calibrar_desde_roi(roi_hsv)
                    print("[INFO] calibrado HSV mano:", piel_inferior, piel_superior)
                else:
                    print("[WARN] ROI muy pequeno")
            else:
                print("[WARN] dibuja un ROI en 'Mano' primero")

        elif tecla == ord('g'):
            if vector_caracteristicas is not None:
                guardar_ejemplo_gesto(vector_caracteristicas, etiqueta_actual)
                if RECOGNIZE_MODE:
                    galeria.append((vector_caracteristicas, etiqueta_actual))
                print(f"[INFO] guardado gesto {etiqueta_actual}")
            else:
                print("[WARN] no hay gesto valido")

        elif tecla in (
            ord('0'),
            ord('1'),
            ord('2'),
            ord('3'),
            ord('4'),
            ord('5'),
            ord('p'),
            ord('-'),
            ord('n'),
        ):
            mapeo = {
                ord('0'): "0dedos",
                ord('1'): "1dedo",
                ord('2'): "2dedos",
                ord('3'): "3dedos",
                ord('4'): "4dedos",
                ord('5'): "5dedos",
                ord('p'): "ok",
                ord('-'): "cool",
                ord('n'): "nook",
            }
            etiqueta_actual = mapeo[tecla]

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
