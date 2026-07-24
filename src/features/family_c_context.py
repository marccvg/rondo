"""Familia C — Contexto partido.

Features:
- context_days_rest_home, away — días desde el último partido de cada equipo (fatiga corta)
- context_days_rest_diff
- context_month_sin, month_cos — estacionalidad
- context_matchday_in_season — posición temporal dentro de temporada (0..1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _long_dates(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["date", "home_team"]].rename(columns={"home_team": "team"})
    away = matches[["date", "away_team"]].rename(columns={"away_team": "team"})
    long = pd.concat([home, away]).sort_values(["team", "date"]).reset_index(drop=True)
    long["days_rest"] = long.groupby("team")["date"].diff().dt.days
    return long.set_index(["team", "date"])


def compute_family_c(matches: pd.DataFrame) -> pd.DataFrame:
    long_idx = _long_dates(matches)
    out = pd.DataFrame(index=matches.index)

    h = long_idx.reindex(list(zip(matches["home_team"], matches["date"])))
    a = long_idx.reindex(list(zip(matches["away_team"], matches["date"])))
    out["context_days_rest_home"] = h["days_rest"].to_numpy()
    out["context_days_rest_away"] = a["days_rest"].to_numpy()
    out["context_days_rest_diff"] = out["context_days_rest_home"] - out["context_days_rest_away"]

    dates = pd.to_datetime(matches["date"])
    out["context_month"] = dates.dt.month.to_numpy()
    out["context_month_sin"] = np.sin(2 * np.pi * out["context_month"] / 12)
    out["context_month_cos"] = np.cos(2 * np.pi * out["context_month"] / 12)

    # matchday_in_season: normalizamos por temporada y liga
    season_min = matches.groupby(["league", "season"])["date"].transform("min")
    season_max = matches.groupby(["league", "season"])["date"].transform("max")
    span_days = (season_max - season_min).dt.days.replace(0, 1)
    out["context_season_progress"] = ((dates - season_min).dt.days / span_days).to_numpy()

    return out


if __name__ == "__main__":
    df = pd.read_parquet("/home/Projects/futbol_elo_poisson/data/processed/matches_all.parquet")
    feats = compute_family_c(df)
    print(f"Family C: {len(feats.columns)} cols")
    print(feats.columns.tolist())
    print(f"\nMissing:\n{feats.isna().sum()}")
    print(f"\nSample:\n{feats.head()}")
