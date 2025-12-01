# storage.py
import os
import time
import json
import numpy as np
from hand_config import GESTURES_DIR

os.makedirs(GESTURES_DIR, exist_ok=True)

def save_gesture_example(feature_vec, label, save_dir=GESTURES_DIR):
    ts = int(time.time() * 1000)
    fp = os.path.join(save_dir, f"{label}_{ts}.npz")
    np.savez(fp, feature=feature_vec, label=label)
    print(f"[INFO] Gesto guardado en {fp}")

def load_gesture_gallery(save_dir=GESTURES_DIR):
    gallery = []
    for fname in os.listdir(save_dir):
        if not fname.endswith(".npz"):
            continue
        data = np.load(os.path.join(save_dir, fname), allow_pickle=True)
        feature = data["feature"]
        label = str(data["label"])
        gallery.append((feature, label))
    print(f"[INFO] Cargadas {len(gallery)} muestras en galería.")
    return gallery

def save_sequence_json(
    acciones,
    out_dir="gestures",
    target_name=None,
    shot_number=None,
    row=None,
    col=None,
):
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time() * 1000)
    payload = {
        "timestamp": ts,
        "sequence": acciones,
    }
    if target_name is not None:
        payload["target"] = target_name
    if shot_number is not None:
        payload["shot"] = shot_number
    if row is not None:
        payload["row"] = row
    if col is not None:
        payload["col"] = col

    if target_name is not None and shot_number is not None:
        fname = f"{target_name}_{shot_number}.json"
    else:
        fname = f"sequence_{ts}.json"

    fp = os.path.join(out_dir, fname)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Secuencia guardada en {fp}")
