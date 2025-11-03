import os
import glob
import time
import json
import numpy as np
from config import SAVE_DIR

def save_gesture_example(feature_vec, label):
    """
    Guarda:
    - feature: vector de características (Hu + radial normalizadas)
    - label: etiqueta del gesto
    en un .npz con timestamp.
    """
    ts = int(time.time() * 1000)
    filepath = os.path.join(SAVE_DIR, f"{label}_{ts}.npz")
    np.savez(filepath, feature=feature_vec, label=label)
    print(f"[INFO] Gesto guardado en {filepath}")

def load_gesture_gallery():
    """
    Carga todos los .npz de la galería
    y devuelve una lista de (feature_vec, label).
    """
    gallery = []
    for fp in glob.glob(os.path.join(SAVE_DIR, "*.npz")):
        data = np.load(fp, allow_pickle=True)
        feature = data["feature"]
        label = str(data["label"])
        gallery.append((feature, label))
    print(f"[INFO] Cargadas {len(gallery)} muestras en galería.")
    return gallery

def save_sequence_json(acciones):
    """
    Guarda la secuencia de acciones (lista de ints/labels numéricos)
    en un JSON con timestamp dentro de SAVE_DIR.

    Formato guardado:
    {
        "timestamp": "2025-10-31_17-22-58",
        "sequence": [5,2,2,4,2]
    }

    Nombre de archivo:
    <timestamp>_seq.json
    """
    if not acciones:
        print("[WARN] Secuencia vacía, no guardo JSON.")
        return None

    ts_struct = time.localtime()
    ts_str = time.strftime("%Y-%m-%d_%H-%M-%S", ts_struct)

    data = {
        "timestamp": ts_str,
        "sequence": acciones
    }

    filename = f"{ts_str}_seq.json"
    filepath = os.path.join(SAVE_DIR, filename)

    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Secuencia guardada en {filepath}")
        return filepath
    except Exception as e:
        print(f"[ERROR] No pude guardar la secuencia en JSON: {e}")
        return None
