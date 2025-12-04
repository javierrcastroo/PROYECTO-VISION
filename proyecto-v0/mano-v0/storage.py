# storage.py
import os
import time
import json
import numpy as np
from hand_config import GESTURES_DIR, ATTACKS_DIR

os.makedirs(GESTURES_DIR, exist_ok=True)
os.makedirs(ATTACKS_DIR, exist_ok=True)


def guardar_ejemplo_gesto(vector_caracteristicas, etiqueta, carpeta_guardado=GESTURES_DIR):
    timestamp = int(time.time() * 1000)
    ruta_archivo = os.path.join(carpeta_guardado, f"{etiqueta}_{timestamp}.npz")
    np.savez(ruta_archivo, feature=vector_caracteristicas, label=etiqueta)
    print(f"[INFO] Gesto guardado en {ruta_archivo}")


def cargar_galeria_gestos(carpeta_guardado=GESTURES_DIR):
    galeria = []
    for nombre_archivo in os.listdir(carpeta_guardado):
        if not nombre_archivo.endswith(".npz"):
            continue
        data = np.load(os.path.join(carpeta_guardado, nombre_archivo), allow_pickle=True)
        vector = data["feature"]
        etiqueta = str(data["label"])
        galeria.append((vector, etiqueta))
    print(f"[INFO] Cargadas {len(galeria)} muestras en galería.")
    return galeria


def guardar_secuencia_json(
    acciones,
    carpeta_salida=ATTACKS_DIR,
    nombre_objetivo=None,
    numero_disparo=None,
    fila=None,
    columna=None,
):
    os.makedirs(carpeta_salida, exist_ok=True)
    timestamp = int(time.time() * 1000)
    payload = {
        "timestamp": timestamp,
        "sequence": acciones,
    }
    if nombre_objetivo is not None:
        payload["target"] = nombre_objetivo
    if numero_disparo is not None:
        payload["shot"] = numero_disparo
    if fila is not None:
        payload["row"] = fila
    if columna is not None:
        payload["col"] = columna

    if nombre_objetivo is not None and numero_disparo is not None:
        nombre_archivo = f"{nombre_objetivo}_{numero_disparo}.json"
    else:
        nombre_archivo = f"sequence_{timestamp}.json"

    ruta_archivo = os.path.join(carpeta_salida, nombre_archivo)
    with open(ruta_archivo, "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2)
    print(f"[INFO] Secuencia guardada en {ruta_archivo}")


def guardar_peticion_reinicio(carpeta_salida=ATTACKS_DIR):
    os.makedirs(carpeta_salida, exist_ok=True)
    payload = {"timestamp": int(time.time()), "action": "restart"}
    ruta_archivo = os.path.join(carpeta_salida, "restart.json")
    with open(ruta_archivo, "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2)
    print(f"[INFO] Peticion de reinicio guardada en {ruta_archivo}")
