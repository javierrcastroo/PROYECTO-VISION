"""Validación sencilla de tableros de Hundir la Flota."""

# Estado global de la partida
GAME_STATE = None


def _build_ships(layout):
    """Crea la lista de barcos a partir de un layout calibrado."""
    ships = []
    ship_two_cells = layout.get("ship_two_cells") or []
    ship_one_cells = layout.get("ship_one_cells") or []

    if ship_two_cells:
        ships.append(set(ship_two_cells))
    for cell in ship_one_cells:
        ships.append({cell})
    return ships


def init_game(player_layouts):
    """Inicializa la partida con las distribuciones de T1 y T2."""
    global GAME_STATE

    players = {}
    for name in ("T1", "T2"):
        layout = player_layouts.get(name, {}) if player_layouts else {}
        ships = _build_ships(layout)
        players[name] = {
            "ships": ships,
            "ship_hits": [set() for _ in ships],
            "cell_states": {cell: "ship" for ship in ships for cell in ship},
            "shots": {},
            "hits": set(),
            "misses": set(),
            "remaining_ships": len(ships),
        }

    GAME_STATE = {
        "players": players,
        "current_attacker": "T1",
        "current_defender": "T2",
        "finished": False,
        "winner": None,
    }
    return GAME_STATE


def _switch_turns():
    """Intercambia atacante y defensor."""
    GAME_STATE["current_attacker"], GAME_STATE["current_defender"] = (
        GAME_STATE["current_defender"],
        GAME_STATE["current_attacker"],
    )


def apply_attack(coord):
    """Aplica un ataque sobre la casilla del defensor actual."""
    if GAME_STATE is None:
        return {"result": "sin_partida"}

    if GAME_STATE.get("finished"):
        return {"result": "fin", "winner": GAME_STATE.get("winner")}

    defender = GAME_STATE["players"][GAME_STATE["current_defender"]]

    if coord in defender["shots"]:
        return {"result": "repetido"}

    if coord not in defender["cell_states"]:
        defender["shots"][coord] = "agua"
        defender["misses"].add(coord)
        _switch_turns()
        return {
            "result": "agua",
            "current_attacker": GAME_STATE["current_attacker"],
            "current_defender": GAME_STATE["current_defender"],
            "game_over": GAME_STATE.get("finished", False),
        }

    defender["shots"][coord] = "tocado"
    defender["hits"].add(coord)
    result = "tocado"

    for idx, ship in enumerate(defender["ships"]):
        if coord in ship:
            defender["ship_hits"][idx].add(coord)
            if defender["ship_hits"][idx] == ship:
                result = "hundido"
                for cell in ship:
                    defender["shots"][cell] = "hundido"
                defender["remaining_ships"] -= 1
                if defender["remaining_ships"] == 0:
                    GAME_STATE["finished"] = True
                    GAME_STATE["winner"] = GAME_STATE["current_attacker"]
            break

    return {
        "result": result,
        "current_attacker": GAME_STATE["current_attacker"],
        "current_defender": GAME_STATE["current_defender"],
        "game_over": GAME_STATE.get("finished", False),
        "winner": GAME_STATE.get("winner"),
    }

def _cells_adjacent(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def evaluate_board(layout):
    """Recibe un layout con listas de celdas ocupadas y devuelve (ok, mensaje)."""
    ship_two_cells = layout.get("ship_two_cells", [])
    ship_one_cells = layout.get("ship_one_cells", [])

    errors = []

    # comprobar cantidades
    if len(ship_two_cells) != 2:
        errors.append("El barco de 2 casillas no está completo")
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
        return True, "Distribución correcta"
    # devolver solo errores únicos para no repetir el mismo texto
    uniq_errors = []
    for err in errors:
        if err not in uniq_errors:
            uniq_errors.append(err)
    return False, " | ".join(uniq_errors)
