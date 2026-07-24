"""
Diagnóstico: qué partidos movieron el ELO de un equipo durante 25/26.
Descompone los +80 de Bayern, los -34 de PSG, los -111 de Liverpool.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/Projects/futbol_elo_poisson")
sys.path.insert(0, str(ROOT / "src"))

from elo import EloState

TARGETS = ["Paris SG", "Bayern Munich", "Liverpool", "Barcelona", "Real Madrid", "Arsenal"]


def main():
    df = pd.read_parquet(ROOT / "data/processed/matches_all.parquet")
    df = df.sort_values("date").reset_index(drop=True)

    # ELO recorriendo todo pero registrando cada delta por partido
    state = EloState()
    history = []
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        rh_before = state.get(home)
        ra_before = state.get(away)
        state.update(home, away, int(row["home_goals"]), int(row["away_goals"]))
        rh_after = state.get(home)
        ra_after = state.get(away)
        delta_h = rh_after - rh_before
        delta_a = ra_after - ra_before
        history.append({
            "date": row["date"], "league": row["league"], "season": row["season"],
            "home": home, "away": away, "gh": row["home_goals"], "ga": row["away_goals"],
            "elo_h_before": rh_before, "elo_a_before": ra_before,
            "delta_h": delta_h, "delta_a": delta_a,
        })

    hist = pd.DataFrame(history)
    season_25 = hist[hist["season"] == "2025-26"]
    print(f"Partidos temporada 25/26 en histórico: {len(season_25):,}")
    print()

    for team in TARGETS:
        rows_home = season_25[season_25["home"] == team].assign(role="H", delta=lambda d: d["delta_h"])
        rows_away = season_25[season_25["away"] == team].assign(role="A", delta=lambda d: d["delta_a"])
        team_rows = pd.concat([rows_home, rows_away]).sort_values("date")

        if team_rows.empty:
            continue

        total_delta = team_rows["delta"].sum()
        wins = ((team_rows["role"]=="H") & (team_rows["gh"] > team_rows["ga"])).sum() + \
               ((team_rows["role"]=="A") & (team_rows["ga"] > team_rows["gh"])).sum()
        draws = (team_rows["gh"] == team_rows["ga"]).sum()
        losses = len(team_rows) - wins - draws

        print(f"═══ {team} — temporada 25/26 ═══")
        print(f"   Partidos: {len(team_rows)}  |  W-D-L: {wins}-{draws}-{losses}  |  ΔELO total: {total_delta:+.1f}")

        # Top 5 partidos que MÁS subieron su ELO
        top_up = team_rows.nlargest(5, "delta")
        # Top 5 partidos que MÁS bajaron
        top_down = team_rows.nsmallest(5, "delta")

        print(f"\n   🟢 TOP 5 GANANCIAS ELO:")
        for _, r in top_up.iterrows():
            opp = r["away"] if r["role"] == "H" else r["home"]
            venue = "local" if r["role"] == "H" else "visitante"
            print(f"      {r['date'].date()}  {team} {r['gh']}-{r['ga']} {opp}  ({venue})  Δ = {r['delta']:+.2f}")

        print(f"\n   🔴 TOP 5 PÉRDIDAS ELO:")
        for _, r in top_down.iterrows():
            opp = r["away"] if r["role"] == "H" else r["home"]
            venue = "local" if r["role"] == "H" else "visitante"
            print(f"      {r['date'].date()}  {team} {r['gh']}-{r['ga']} {opp}  ({venue})  Δ = {r['delta']:+.2f}")
        print()


if __name__ == "__main__":
    main()
