"""Familia F — Matchup (interacción entre equipos).

Features:
- matchup_h2h_home_wins_prev5, home_win_pct_prev5 — historial directo últimos 5 encuentros
- matchup_h2h_avg_goals_home, avg_goals_away
- matchup_h2h_avg_total_goals
"""
from __future__ import annotations

import pandas as pd


def compute_family_f(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Para cada partido, mira los últimos `window` encuentros entre estos 2 equipos
    (independiente de quién jugó como local).
    """
    matches = matches.reset_index(drop=True).copy()
    matches["pair_key"] = matches.apply(
        lambda r: "|".join(sorted([r["home_team"], r["away_team"]])), axis=1
    )
    matches["_orig_idx"] = matches.index

    # ordenar por pair_key + date para iterar
    sorted_df = matches.sort_values(["pair_key", "date"]).reset_index(drop=True)

    n = len(sorted_df)
    hh_wins = [None] * n
    aa_wins = [None] * n
    draws = [None] * n
    total_goals = [None] * n

    # Para cada grupo pair_key
    for pair, group in sorted_df.groupby("pair_key", sort=False):
        idxs = group.index.tolist()
        for i, row_i in enumerate(idxs):
            prev = idxs[max(0, i - window):i]
            if not prev:
                continue
            current_home = sorted_df.at[row_i, "home_team"]
            prev_rows = sorted_df.loc[prev]
            # Contamos wins desde perspectiva del ACTUAL home team
            wins_current_home = ((prev_rows["home_team"] == current_home) & (prev_rows["result"] == "H")).sum() \
                                + ((prev_rows["away_team"] == current_home) & (prev_rows["result"] == "A")).sum()
            wins_current_away = ((prev_rows["home_team"] == current_home) & (prev_rows["result"] == "A")).sum() \
                                + ((prev_rows["away_team"] == current_home) & (prev_rows["result"] == "H")).sum()
            draws_count = (prev_rows["result"] == "D").sum()
            tgoals = (prev_rows["home_goals"] + prev_rows["away_goals"]).mean()
            hh_wins[row_i] = wins_current_home
            aa_wins[row_i] = wins_current_away
            draws[row_i] = draws_count
            total_goals[row_i] = tgoals

    sorted_df["matchup_h2h_wins_currhome"] = hh_wins
    sorted_df["matchup_h2h_wins_curraway"] = aa_wins
    sorted_df["matchup_h2h_draws"] = draws
    sorted_df["matchup_h2h_avg_total_goals"] = total_goals

    # Recuperar orden original
    result = sorted_df.set_index("_orig_idx").sort_index()
    out = pd.DataFrame(index=matches.index)
    out["matchup_h2h_wins_currhome"] = result["matchup_h2h_wins_currhome"].to_numpy()
    out["matchup_h2h_wins_curraway"] = result["matchup_h2h_wins_curraway"].to_numpy()
    out["matchup_h2h_draws"] = result["matchup_h2h_draws"].to_numpy()
    out["matchup_h2h_avg_total_goals"] = result["matchup_h2h_avg_total_goals"].to_numpy()
    total_prev = out[["matchup_h2h_wins_currhome", "matchup_h2h_wins_curraway", "matchup_h2h_draws"]].sum(axis=1)
    out["matchup_h2h_home_winpct"] = (out["matchup_h2h_wins_currhome"] / total_prev).where(total_prev > 0)
    return out


if __name__ == "__main__":
    df = pd.read_parquet("/home/Projects/futbol_elo_poisson/data/processed/matches_all.parquet")
    feats = compute_family_f(df)
    print(f"Family F: {len(feats.columns)} cols")
    print(feats.columns.tolist())
    print(f"Non-null pct: {(1 - feats.isna().mean()) * 100}")
