"""Familia B — Forma reciente.

Features:
- form_pts_r5, r10 — puntos rolling últimos N partidos por equipo
- form_pts_r5_decay — puntos ponderados con decay exponencial ([1.0, 0.8, 0.64, 0.51, 0.41])
- form_win_streak, form_unbeaten_streak — rachas actuales
- form_diff — dif de forma entre home y away

Todo con shift(1) para no leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _points_for_team(row, team_col: str) -> int:
    if row["result"] == "H":
        return 3 if team_col == "home_team" else 0
    if row["result"] == "A":
        return 3 if team_col == "away_team" else 0
    return 1  # empate


def _long_form(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches.apply(lambda r: pd.Series({
        "date": r["date"], "team": r["home_team"],
        "points": _points_for_team(r, "home_team"),
        "win": int(r["result"] == "H"),
        "unbeaten": int(r["result"] in {"H", "D"}),
    }), axis=1)
    away = matches.apply(lambda r: pd.Series({
        "date": r["date"], "team": r["away_team"],
        "points": _points_for_team(r, "away_team"),
        "win": int(r["result"] == "A"),
        "unbeaten": int(r["result"] in {"A", "D"}),
    }), axis=1)
    long = pd.concat([home, away]).sort_values(["team", "date"]).reset_index(drop=True)
    return long


def _current_streak(series: pd.Series) -> pd.Series:
    """Cuenta la racha actual: cuántos 1 consecutivos previos al índice i (excluye i)."""
    shifted = series.shift(1).fillna(0)
    # Group by broken streaks (marker every time shifted == 0)
    streak = shifted.groupby((shifted == 0).cumsum()).cumsum()
    return streak.astype(int)


def compute_family_b(matches: pd.DataFrame, decay_alpha: float = 0.8) -> pd.DataFrame:
    long = _long_form(matches)

    long["pts_r5"] = long.groupby("team")["points"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )
    long["pts_r10"] = long.groupby("team")["points"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).sum()
    )

    # Decay exponencial: peso_i = alpha^i
    weights = np.array([decay_alpha ** i for i in range(5)])
    def wsum(s):
        arr = s.shift(1).rolling(5, min_periods=1).apply(
            lambda w: np.dot(w[::-1][: len(w)], weights[: len(w)]) / weights[: len(w)].sum(),
            raw=True,
        )
        return arr
    long["pts_r5_decay"] = long.groupby("team")["points"].transform(wsum)

    long["win_streak"] = long.groupby("team")["win"].transform(_current_streak)
    long["unbeaten_streak"] = long.groupby("team")["unbeaten"].transform(_current_streak)

    long_idx = long.set_index(["team", "date"])

    out = pd.DataFrame(index=matches.index)
    for side, tcol in [("home", "home_team"), ("away", "away_team")]:
        joined = long_idx.reindex(list(zip(matches[tcol], matches["date"])))
        out[f"form_pts_r5_{side}"] = joined["pts_r5"].to_numpy()
        out[f"form_pts_r10_{side}"] = joined["pts_r10"].to_numpy()
        out[f"form_pts_r5_decay_{side}"] = joined["pts_r5_decay"].to_numpy()
        out[f"form_win_streak_{side}"] = joined["win_streak"].to_numpy()
        out[f"form_unbeaten_streak_{side}"] = joined["unbeaten_streak"].to_numpy()

    out["form_pts_diff_r5"] = out["form_pts_r5_home"] - out["form_pts_r5_away"]
    out["form_pts_diff_r10"] = out["form_pts_r10_home"] - out["form_pts_r10_away"]
    return out


if __name__ == "__main__":
    df = pd.read_parquet("/home/Projects/futbol_elo_poisson/data/processed/matches_all.parquet")
    feats = compute_family_b(df)
    print(f"Family B: {len(feats.columns)} cols, {len(feats):,} filas")
    print(feats.columns.tolist())
    print(f"\nSample non-NaN:\n{feats.dropna().head()}")
