# board_state.py
"""
Gestión del estado de cada tablero + origen global compartido
"""

def inicializar_estado_tablero(nombre):
    """
    Crea un diccionario con el estado de un tablero concreto (T1, T2, ...).
    """
    return {
        "name": nombre,
        "last_quad": None,
        "miss": 0,
        "cm_per_pix": None,
        "ship_two_cells": [],
        "ship_one_cells": [],
        "ship_two_points": [],
        "ship_one_points": [],
        "attacked_cells": set(),
        "hits": set(),
        "misses": set(),
    }


# origen global = el cubo verde que el usuario calibra con 'r'
# lo vamos actualizando cada frame
ORIGEN_GLOBAL = None       # (x, y) en píxeles
ORIGEN_GLOBAL_FALLOS = 0   # para aguantar unos frames si desaparece
ORIGEN_GLOBAL_MAX_FALLOS = 10
