"""Validacion sencilla de tableros de Hundir la Flota y gestion de turnos."""

BOARD_LABELS = ["A", "B", "C", "D", "E"]

def _cells_adjacent(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _normalize_cell(cell):
    try:
        row = int(cell[0])
    except (TypeError, ValueError, IndexError):
        row = cell[0]
    try:
        col = int(cell[1])
    except (TypeError, ValueError, IndexError):
        col = cell[1]
    return (row, col)


def _format_cell_label(cell):
    row, col = _normalize_cell(cell)
    if isinstance(row, int) and isinstance(col, int):
        return f"{chr(ord('A') + col)}{row + 1}"
    return f"{row},{col}"


def evaluate_board(layout):
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
                if _cells_adjacent(cell, other):
                    errors.append("Los barcos no pueden tocarse")
                    break
    for idx, cell in enumerate(singles):
        for other in singles[idx + 1:]:
            if _cells_adjacent(cell, other):
                errors.append("Los barcos no pueden tocarse")
                break
        if errors:
            break

    if not errors:
        return True, "Distribucion correcta"
    # devolver solo errores únicos para no repetir el mismo texto
    uniq_errors = []
    for err in errors:
        if err not in uniq_errors:
            uniq_errors.append(err)
    return False, " | ".join(uniq_errors)


def init_game_state(layouts_by_name):
    boards = {}
    for name, layout in layouts_by_name.items():
        boards[name] = _build_board_from_layout(name, layout)
    return {
        "boards": boards,
        "current_attacker": "T1",
        "current_defender": "T2",
        "finished": False,
        "winner": None,
    }


def apply_attack(game_state, target_board, row, col):
    if game_state.get("finished"):
        return {"status": "finished", "winner": game_state.get("winner")}

    attacker = game_state.get("current_attacker")
    defender = game_state.get("current_defender")

    if target_board != defender:
        return {
            "status": "wrong_target",
            "attacker": attacker,
            "defender": defender,
            "cell": _format_cell(row, col),
        }

    board = game_state["boards"].get(defender)
    if board is None:
        return None

    row_i = int(row)
    col_i = int(col)
    cell = (row_i, col_i)
    if row_i < 0 or col_i < 0 or row_i >= board["board_size"] or col_i >= board["board_size"]:
        return {
            "status": "invalid",
            "attacker": attacker,
            "defender": defender,
            "cell": _format_cell(row, col),
        }
    if cell in board["attacked_cells"]:
        return {
            "status": "invalid",
            "attacker": attacker,
            "defender": defender,
            "cell": _format_cell(row, col),
        }

    board["attacked_cells"].add(cell)

    if cell not in board["cell_to_ship"]:
        _switch_turn(game_state)
        return {
            "status": "agua",
            "attacker": attacker,
            "defender": defender,
            "cell": _format_cell(row, col),
        }

    ship_id = board["cell_to_ship"][cell]
    ship = board["ships"][ship_id]
    ship["hits"].add(cell)

    if len(ship["hits"]) >= len(ship["cells"]):
        board["ships_alive"] -= 1
        winner = None
        if board["ships_alive"] <= 0:
            game_state["finished"] = True
            winner = attacker
            game_state["winner"] = winner
        return {
            "status": "hundido",
            "attacker": attacker,
            "defender": defender,
            "cell": _format_cell(row, col),
            "winner": winner,
        }

    return {
        "status": "tocado",
        "attacker": attacker,
        "defender": defender,
        "cell": _format_cell(row, col),
    }


def _build_board_from_layout(name, layout):
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


def _switch_turn(game_state):
    atk = game_state.get("current_attacker")
    defn = game_state.get("current_defender")
    game_state["current_attacker"], game_state["current_defender"] = defn, atk


def _format_cell(row, col):
    col = int(col)
    row = int(row)
    col_label = BOARD_LABELS[col] if 0 <= col < len(BOARD_LABELS) else str(col)
    return f"{col_label}{row + 1}"
