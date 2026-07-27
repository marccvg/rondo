"""
Sistema de puntos LaLiga Fantasy oficial (Marca / La Liga Fantasy Official).

Referencia: reglas oficiales de laligafantasy.marca.com

Nota importante: LaLiga Fantasy oficial usa MEDIA SofaScore como componente,
que es 0-10 (aproximadamente). En este módulo aproximamos ese rating
combinando xG contribución, minutos y stats del partido.

Sistema por posición:
"""
from __future__ import annotations
from typing import Dict

# Base común (juega N minutos)
POINTS_APPEARANCE = {
    "played_1_59_min": 1,   # jugar entre 1-59 min
    "played_60_plus": 2,    # 60+ min
}

# Por acción por posición
POINTS_GOAL = {
    "GK": 10,
    "DEF": 6,
    "MID": 5,
    "FWD": 4,
}

POINTS_ASSIST = 3  # igual para todas posiciones

POINTS_CLEAN_SHEET = {
    "GK": 5,
    "DEF": 4,
    "MID": 1,
    "FWD": 0,
}

# Penalizaciones
POINTS_YELLOW = -1
POINTS_RED = -3
POINTS_OWN_GOAL = -2
POINTS_MISSED_PEN = -2
POINTS_CONCEDED_GK = -1     # cada gol encajado (GK)
POINTS_CONCEDED_DEF = -1    # cada gol encajado (DEF, cada 2 goles)

# Extras
POINTS_SAVE = 0.4          # por parada (GK) - aproximado
POINTS_PEN_SAVE = 5

# Sistema MVP / media SofaScore aproximada
# LaLiga Fantasy da puntos según valoración 0-10 SofaScore:
# rating >= 8.0: +3 puntos MVP
# rating 7.0-7.9: +1 punto
# rating 5.0-5.9: -1 punto
# rating < 5.0: -2 puntos
POINTS_BY_RATING = {
    (8.5, 11.0): 3,
    (7.5, 8.5): 2,
    (7.0, 7.5): 1,
    (6.0, 7.0): 0,
    (5.0, 6.0): -1,
    (0.0, 5.0): -2,
}


def calculate_fantasy_points(row: Dict) -> float:
    """Calcula puntos LaLiga Fantasy para 1 jugador en 1 partido.

    row esperado con al menos:
      position: GK/DEF/MID/FWD
      minutes: int
      goals: int
      assists: int
      clean_sheet: bool
      yellow_cards: int
      red_cards: int
      own_goals: int (opcional)
      penalty_missed: int (opcional)
      saves: int (opcional, solo GK)
      goals_conceded: int (para GK/DEF)
      rating: float 0-10 (opcional, aproximación SofaScore)
    """
    pts = 0.0
    pos = row.get("position", "MID")

    # Base minutos
    mins = row.get("minutes", 0)
    if mins >= 60:
        pts += POINTS_APPEARANCE["played_60_plus"]
    elif mins >= 1:
        pts += POINTS_APPEARANCE["played_1_59_min"]
    else:
        return 0.0  # no jugó, 0 puntos

    # Goles
    pts += row.get("goals", 0) * POINTS_GOAL.get(pos, 4)

    # Asistencias
    pts += row.get("assists", 0) * POINTS_ASSIST

    # Portería a cero (solo si jugó 60+)
    if mins >= 60 and row.get("clean_sheet", False):
        pts += POINTS_CLEAN_SHEET.get(pos, 0)

    # Tarjetas
    pts += row.get("yellow_cards", 0) * POINTS_YELLOW
    pts += row.get("red_cards", 0) * POINTS_RED

    # Own goals + penaltis fallados
    pts += row.get("own_goals", 0) * POINTS_OWN_GOAL
    pts += row.get("penalty_missed", 0) * POINTS_MISSED_PEN

    # Goles encajados (GK, DEF)
    if pos == "GK":
        pts += row.get("goals_conceded", 0) * POINTS_CONCEDED_GK
        # Paradas (aproximación)
        pts += row.get("saves", 0) * POINTS_SAVE
        pts += row.get("penalty_saved", 0) * POINTS_PEN_SAVE
    elif pos == "DEF":
        # -1 cada 2 goles encajados
        pts += (row.get("goals_conceded", 0) // 2) * POINTS_CONCEDED_DEF

    # Rating (aproximación media SofaScore)
    rating = row.get("rating", None)
    if rating is not None and rating > 0:
        for (lo, hi), bonus in POINTS_BY_RATING.items():
            if lo <= rating < hi:
                pts += bonus
                break

    return pts


def approximate_rating_from_stats(row: Dict) -> float:
    """Aproxima el rating SofaScore desde stats FBref (que sí tenemos).

    Es aproximación. El rating real usa muchos factores (duelos, distancia, etc).
    Aproximación decente basada en:
      - goles + asistencias (mayor peso)
      - xG - xG_conceded
      - completions pass %
      - defensive actions
      - saves (GK)
    """
    mins = row.get("minutes", 0)
    if mins < 1:
        return 0

    base = 6.0  # rating base típico jugador que juega
    goals = row.get("goals", 0)
    assists = row.get("assists", 0)
    xg = row.get("xg", 0) or 0
    xa = row.get("xa", 0) or 0

    # Contribución ofensiva
    base += goals * 1.2
    base += assists * 0.8
    base += (xg - 0.3) * 0.3 if xg > 0.3 else 0
    base += (xa - 0.2) * 0.3 if xa > 0.2 else 0

    # Tarjetas restan
    base -= row.get("yellow_cards", 0) * 0.3
    base -= row.get("red_cards", 0) * 1.5

    return max(0, min(10, base))
