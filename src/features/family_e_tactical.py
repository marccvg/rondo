"""Familia E — Estilo táctico + xG (basado en FBref schedule).

Features:
- tactical_xg_home_r5, r10 — media xG generada últimos N partidos
- tactical_xga_home_r5, r10 — media xG concedida últimos N
- tactical_xg_overperform_home_r10 — (goles reales - xG) rolling → regresión a media
- tactical_xg_diff_r10 — diferencia xG - xGA rolling (fuerza real)

Todas con shift(1) para evitar leak.
Espera columnas en input: date, home_team, away_team, xg_home, xg_away, home_goals, away_goals.
"""
from __future__ import annotations

import pandas as pd


def _long_tactical(matches: pd.DataFrame) -> pd.DataFrame:
    """Desdobla home/away en formato largo con xG generado y concedido por equipo-partido."""
    home = matches[["date", "home_team", "xg_home", "xg_away", "home_goals", "away_goals"]].rename(
        columns={
            "home_team": "team",
            "xg_home": "xg_scored",
            "xg_away": "xg_conceded",
            "home_goals": "goals_scored",
            "away_goals": "goals_conceded",
        }
    )
    away = matches[["date", "away_team", "xg_away", "xg_home", "away_goals", "home_goals"]].rename(
        columns={
            "away_team": "team",
            "xg_away": "xg_scored",
            "xg_home": "xg_conceded",
            "away_goals": "goals_scored",
            "home_goals": "goals_conceded",
        }
    )
    long = pd.concat([home, away]).sort_values(["team", "date"]).reset_index(drop=True)
    return long


def compute_family_e(matches: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
    """
    matches debe incluir xg_home, xg_away (de FBref schedule).
    Devuelve DataFrame same length.
    """
    required = {"date", "home_team", "away_team", "xg_home", "xg_away", "home_goals", "away_goals"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    long = _long_tactical(matches)

    for w in windows:
        long[f"xg_scored_r{w}"] = long.groupby("team")["xg_scored"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        long[f"xg_conceded_r{w}"] = long.groupby("team")["xg_conceded"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        long[f"xg_diff_r{w}"] = long[f"xg_scored_r{w}"] - long[f"xg_conceded_r{w}"]
        # Overperformance (goles - xG): positivo = "suerte" reciente
        long[f"xg_overperform_r{w}"] = long.groupby("team").apply(
            lambda g: (g["goals_scored"] - g["xg_scored"]).shift(1).rolling(w, min_periods=1).mean(),
            include_groups=False,
        ).reset_index(level=0, drop=True)

    long_idx = long.set_index(["team", "date"])
    out = pd.DataFrame(index=matches.index)

    for side, tcol in [("home", "home_team"), ("away", "away_team")]:
        joined = long_idx.reindex(list(zip(matches[tcol], matches["date"])))
        for w in windows:
            out[f"tactical_xg_scored_{side}_r{w}"] = joined[f"xg_scored_r{w}"].to_numpy()
            out[f"tactical_xg_conceded_{side}_r{w}"] = joined[f"xg_conceded_r{w}"].to_numpy()
            out[f"tactical_xg_diff_{side}_r{w}"] = joined[f"xg_diff_r{w}"].to_numpy()
            out[f"tactical_xg_overperform_{side}_r{w}"] = joined[f"xg_overperform_r{w}"].to_numpy()

    # Interacciones útiles
    for w in windows:
        out[f"tactical_xg_diff_rel_r{w}"] = (
            out[f"tactical_xg_diff_home_r{w}"] - out[f"tactical_xg_diff_away_r{w}"]
        )
    return out


if __name__ == "__main__":
    # Test con datos dummy
    import numpy as np
    n = 100
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "home_team": ["A", "B"] * (n // 2),
        "away_team": ["B", "A"] * (n // 2),
        "xg_home": np.random.uniform(0.5, 2.5, n),
        "xg_away": np.random.uniform(0.5, 2.5, n),
        "home_goals": np.random.poisson(1.5, n),
        "away_goals": np.random.poisson(1.2, n),
    })
    feats = compute_family_e(df)
    print(f"Family E: {len(feats.columns)} cols")
    print(feats.columns.tolist())
