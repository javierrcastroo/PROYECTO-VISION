import numpy as np
from collections import Counter

def knn_predict(feature_vec, gallery, k=5):
    """
    Clasificador k-NN con voto mayoritario.
    - gallery: lista de (feature_vec_guardado, label)
    Devuelve (label_predicho, dist_media_vecinos).
    """
    if len(gallery) == 0:
        return None, None

    dists = []
    for f_ref, lab in gallery:
        # seguridad: solo comparamos vectores con misma dim
        if f_ref.shape != feature_vec.shape:
            continue
        dist = np.linalg.norm(feature_vec - f_ref)
        dists.append((dist, lab))

    if len(dists) == 0:
        return None, None

    dists.sort(key=lambda x: x[0])
    k_neigh = dists[:k]

    labels = [lab for (d, lab) in k_neigh]
    most_common_label, _ = Counter(labels).most_common(1)[0]

    avg_dist = float(np.mean([d for (d, _) in k_neigh]))

    return most_common_label, avg_dist
