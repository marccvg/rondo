"""Familia A — Fuerza base.

Features:
- strength_elo_home, strength_elo_away (ratings pre-partido, ya en input)
- strength_elo_diff
- strength_elo_prob_home (probabilidad esperada según ELO)
- strength_goals_scored_rolling_N (media goles marcados rolling N partidos previos)
- strength_goals_conceded_rolling_N
- strength_goal_diff_rolling_N
- strength_home_win_pct_home_venue (ventaja específica local en su estadio)

Rolling windows: 5, 10, 20.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_team_stats(matches: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    """
    Para cada equipo, calcula media goles marcados y encajados en últimos N partidos
    (venue-agnostic, cuenta local+visitante). Usa shift(1) para no leak.

    Devuelve DataFrame indexado (team, date) con columnas por ventana.
    """
    # "Long format": una fila por (equipo, partido) — desdoblamos home/away
    home_rows = matches[["date", "home_team", "home_goals", "away_goals"]].rename(
        columns={"home_team": "team", "home_goals": "goals_scored", "away_goals": "goals_conceded"}
    )
    away_rows = matches[["date", "away_team", "away_goals", "home_goals"]].rename(
        columns={"away_team": "team", "away_goals": "goals_scored", "home_goals": "goals_conceded"}
    )
    long = pd.concat([home_rows, away_rows]).sort_values(["team", "date"]).reset_index(drop=True)

    for w in windows:
        long[f"goals_scored_r{w}"] = (
            long.groupby("team")["goals_scored"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
        long[f"goals_conceded_r{w}"] = (
            long.groupby("team")["goals_conceded"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
    return long.set_index(["team", "date"])


def compute_family_a(matches: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    """
    Espera columnas: date, home_team, away_team, home_goals, away_goals, elo_home, elo_away, elo_prob_home.
    Devuelve DataFrame mismo orden con features familia A.
    """
    from elo import expected_score

    long_stats = _rolling_team_stats(matches, windows=windows)
    out = pd.DataFrame(index=matches.index)
    out["strength_elo_home"] = matches["elo_home"].to_numpy()
    out["strength_elo_away"] = matches["elo_away"].to_numpy()
    out["strength_elo_diff"] = matches["elo_home"].to_numpy() - matches["elo_away"].to_numpy()
    out["strength_elo_prob_home"] = matches["elo_prob_home"].to_numpy()

    for w in windows:
        # join por (home_team, date) para features del local
        h = long_stats.reindex(
            list(zip(matches["home_team"], matches["date"]))
        )
        a = long_stats.reindex(
            list(zip(matches["away_team"], matches["date"]))
        )
        out[f"strength_home_gs_r{w}"] = h[f"goals_scored_r{w}"].to_numpy()
        out[f"strength_home_gc_r{w}"] = h[f"goals_conceded_r{w}"].to_numpy()
        out[f"strength_away_gs_r{w}"] = a[f"goals_scored_r{w}"].to_numpy()
        out[f"strength_away_gc_r{w}"] = a[f"goals_conceded_r{w}"].to_numpy()
        out[f"strength_gd_diff_r{w}"] = (
            (out[f"strength_home_gs_r{w}"] - out[f"strength_home_gc_r{w}"])
            - (out[f"strength_away_gs_r{w}"] - out[f"strength_away_gc_r{w}"])
        )
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from elo import compute_elo_history

    df = pd.read_parquet("/home/Projects/futbol_elo_poisson/data/processed/matches_all.parquet")
    df = compute_elo_history(df)
    feats = compute_family_a(df)
    print(f"Family A features: {len(feats.columns)} columnas, {len(feats):,} filas")
    print(f"Columnas: {feats.columns.tolist()}")
    print(f"\nMissing por columna:\n{feats.isna().sum()}")
    print(f"\nSample:\n{feats.head()}")
