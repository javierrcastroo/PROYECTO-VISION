"""Validacion sencilla de tableros de Hundir la Flota y gestion de turnos."""

ETIQUETAS_TABLERO = ["A", "B", "C", "D", "E"]


def _celdas_adyacentes(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _normalizar_celda(cell):
    try:
        row = int(cell[0])
    except (TypeError, ValueError, IndexError):
        row = cell[0]
    try:
        col = int(cell[1])
    except (TypeError, ValueError, IndexError):
        col = cell[1]
    return (row, col)


def _formatear_etiqueta_celda(cell):
    row, col = _normalizar_celda(cell)
    if isinstance(row, int) and isinstance(col, int):
        return f"{chr(ord('A') + col)}{row + 1}"
    return f"{row},{col}"


def evaluar_tablero(layout):
    """Recibe un layout con listas de celdas ocupadas y devuelve (ok, mensaje)."""
    ship_two_cells = layout.get("ship_two_cells", [])
    ship_one_cells = layout.get("ship_one_cells", [])

    errors = []

    # comprobar cantidades
    if len(ship_two_cells) != 2:
        errors.append("El barco de 2 casillas no esta completo")
    if len(ship_one_cells) != 3:
        errors.append("Tiene que haber exactamente 3 barcos de 1 casilla")

    # comprobar duplicados
    all_cells = ship_two_cells + ship_one_cells
    if len(all_cells) != len(set(all_cells)):
        errors.append("Hay celdas solapadas entre barcos")

    # comprobar barco de tamaño 2
    if len(ship_two_cells) == 2:
        (r0, c0), (r1, c1) = ship_two_cells
        dr = abs(r0 - r1)
        dc = abs(c0 - c1)
        if not ((dr == 0 and dc == 1) or (dc == 0 and dr == 1)):
            errors.append("El barco de 2 debe estar recto y ocupar 2 casillas contiguas")

    # comprobar separaciones
    singles = ship_one_cells
    if ship_two_cells:
        for cell in ship_two_cells:
            for other in singles:
                if _celdas_adyacentes(cell, other):
                    errors.append("Los barcos no pueden tocarse")
                    break
    for idx, cell in enumerate(singles):
        for other in singles[idx + 1:]:
            if _celdas_adyacentes(cell, other):
                errors.append("Los barcos no pueden tocarse")
                break
        if errors:
            break

    if not errors:
        return True, "Distribucion correcta"
    # devolver solo errores únicos para no repetir el mismo texto
    errores_unicos = []
    for err in errors:
        if err not in errores_unicos:
            errores_unicos.append(err)
    return False, " | ".join(errores_unicos)


def inicializar_estado_partida(layouts_por_nombre):
    tableros = {}
    for name, layout in layouts_por_nombre.items():
        tableros[name] = _construir_tablero_desde_layout(name, layout)
    return {
        "boards": tableros,
        "current_attacker": "T1",
        "current_defender": "T2",
        "finished": False,
        "winner": None,
    }


def aplicar_ataque(estado_partida, tablero_objetivo, fila, columna):
    if estado_partida.get("finished"):
        return {"status": "finished", "winner": estado_partida.get("winner")}

    atacante = estado_partida.get("current_attacker")
    defensor = estado_partida.get("current_defender")

    if tablero_objetivo != defensor:
        return {
            "status": "wrong_target",
            "attacker": atacante,
            "defender": defensor,
            "cell": _formatear_celda(fila, columna),
        }

    board = estado_partida["boards"].get(defensor)
    if board is None:
        return None

    row_i = int(fila)
    col_i = int(columna)
    cell = (row_i, col_i)
    if row_i < 0 or col_i < 0 or row_i >= board["board_size"] or col_i >= board["board_size"]:
        return {
            "status": "invalid",
            "attacker": atacante,
            "defender": defensor,
            "cell": _formatear_celda(fila, columna),
        }
    if cell in board["attacked_cells"]:
        return {
            "status": "invalid",
            "attacker": atacante,
            "defender": defensor,
            "cell": _formatear_celda(fila, columna),
        }

    board["attacked_cells"].add(cell)

    if cell not in board["cell_to_ship"]:
        _cambiar_turno(estado_partida)
        return {
            "status": "agua",
            "attacker": atacante,
            "defender": defensor,
            "cell": _formatear_celda(fila, columna),
        }

    ship_id = board["cell_to_ship"][cell]
    ship = board["ships"][ship_id]
    ship["hits"].add(cell)

    if len(ship["hits"]) >= len(ship["cells"]):
        board["ships_alive"] -= 1
        winner = None
        if board["ships_alive"] <= 0:
            estado_partida["finished"] = True
            winner = atacante
            estado_partida["winner"] = winner
        return {
            "status": "hundido",
            "attacker": atacante,
            "defender": defensor,
            "cell": _formatear_celda(fila, columna),
            "winner": winner,
        }

    return {
        "status": "tocado",
        "attacker": atacante,
        "defender": defensor,
        "cell": _formatear_celda(fila, columna),
    }


def _construir_tablero_desde_layout(name, layout):
    cell_to_ship = {}
    ships = {}

    ship_two = layout.get("ship_two_cells", [])
    ship_one = layout.get("ship_one_cells", [])

    if ship_two:
        ship_id = f"{name}_B2"
        ships[ship_id] = {"cells": set(ship_two), "hits": set()}
        for cell in ship_two:
            cell_to_ship[tuple(cell)] = ship_id

    for idx, cell in enumerate(ship_one):
        ship_id = f"{name}_B1_{idx}"
        ships[ship_id] = {"cells": {tuple(cell)}, "hits": set()}
        cell_to_ship[tuple(cell)] = ship_id

    return {
        "name": name,
        "cell_to_ship": cell_to_ship,
        "ships": ships,
        "ships_alive": len(ships),
        "attacked_cells": set(),
        "board_size": layout.get("board_size", 5),
    }


def _cambiar_turno(estado_partida):
    atk = estado_partida.get("current_attacker")
    defn = estado_partida.get("current_defender")
    estado_partida["current_attacker"], estado_partida["current_defender"] = defn, atk


def _formatear_celda(row, col):
    col = int(col)
    row = int(row)
    etiqueta_columna = ETIQUETAS_TABLERO[col] if 0 <= col < len(ETIQUETAS_TABLERO) else str(col)
    return f"{etiqueta_columna}{row + 1}"
