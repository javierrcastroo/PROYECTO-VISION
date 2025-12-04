# classifier.py
import numpy as np


def predecir_knn(vector_caracteristicas, galeria, k=5):
    """
    galeria: lista de (vector_caracteristicas_guardado, etiqueta)
    """
    if len(galeria) == 0:
        return None, None

    distancias = []
    for vector_referencia, etiqueta in galeria:
        if vector_referencia.shape != vector_caracteristicas.shape:
            continue
        distancia = np.linalg.norm(vector_caracteristicas - vector_referencia)
        distancias.append((distancia, etiqueta))

    if len(distancias) == 0:
        return None, None

    distancias.sort(key=lambda x: x[0])
    k = min(k, len(distancias))
    vecinos = distancias[:k]

    # voto mayoritario
    etiquetas = [etiqueta for _, etiqueta in vecinos]
    # distancia media
    distancia_media = sum(d for d, _ in vecinos) / k

    # escoger la etiqueta más frecuente
    mejor_etiqueta = max(set(etiquetas), key=etiquetas.count)
    return mejor_etiqueta, distancia_media
