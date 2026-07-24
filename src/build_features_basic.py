"""Consolida familias A, B, C, F en una única tabla features_basic.parquet."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from elo import compute_elo_history
from features.family_a_strength import compute_family_a
from features.family_b_form import compute_family_b
from features.family_c_context import compute_family_c
from features.family_f_matchup import compute_family_f

ROOT = Path("/home/Projects/futbol_elo_poisson")
PROC = ROOT / "data" / "processed"


def main():
    df = pd.read_parquet(PROC / "matches_all.parquet")
    print(f"Base: {len(df):,} partidos")
    df = compute_elo_history(df)

    print("Family A...")
    fa = compute_family_a(df)
    print(f"Family B...")
    fb = compute_family_b(df)
    print(f"Family C...")
    fc = compute_family_c(df)
    print(f"Family F...")
    ff = compute_family_f(df)

    # Concatena en el orden de df (los índices coinciden)
    features = pd.concat([fa, fb, fc, ff], axis=1)
    print(f"Features totales: {len(features.columns)} cols")

    # Añade target + metadata
    features["target_goals_home"] = df["home_goals"].to_numpy()
    features["target_goals_away"] = df["away_goals"].to_numpy()
    features["target_result"] = df["result"].to_numpy()
    features["meta_date"] = df["date"].to_numpy()
    features["meta_league"] = df["league"].to_numpy()
    features["meta_season"] = df["season"].to_numpy()
    features["meta_home_team"] = df["home_team"].to_numpy()
    features["meta_away_team"] = df["away_team"].to_numpy()
    features["meta_pinnacle_home"] = df["pinnacle_home"].to_numpy()
    features["meta_pinnacle_draw"] = df["pinnacle_draw"].to_numpy()
    features["meta_pinnacle_away"] = df["pinnacle_away"].to_numpy()

    out = PROC / "features_basic.parquet"
    features.to_parquet(out)
    print(f"Guardado: {out}")
    print(f"Missing por familia:")
    for prefix in ["strength_", "form_", "context_", "matchup_"]:
        cols = [c for c in features.columns if c.startswith(prefix)]
        miss = features[cols].isna().mean().mean() * 100
        print(f"  {prefix:12s}: {len(cols):2d} cols, {miss:.1f}% missing global")


if __name__ == "__main__":
    main()
