"""
ELO con K adaptativo por competición (approach FiveThirtyEight / ClubELO).

K = 20 base para ligas domésticas
K × 1.5 = 30 para Conference League
K × 2.0 = 40 para Europa League
K × 2.5 = 50 para Champions League

Rationale: los partidos cross-liga aportan MUCHA más información que los domésticos
(un equipo que gana en Champions demuestra fuerza real, uno que gana en su liga
puede estar aprovechando canibalización local).
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from elo import expected_score, goal_diff_multiplier, DEFAULT_INITIAL, DEFAULT_HOME_ADV

ROOT = Path("/home/Projects/futbol_elo_poisson")
PROC = ROOT / "data" / "processed"

# K por competición (multiplicadores sobre K_base=20)
K_BY_COMPETITION = {
    "champions_league":     50.0,  # x2.5
    "europa_league":        40.0,  # x2.0
    "conference_league":    30.0,  # x1.5
    # Ligas domésticas: K estándar = 20
    "domestic_premier":     20.0,
    "domestic_laliga":      20.0,
    "domestic_seriea":      20.0,
    "domestic_bundesliga":  20.0,
    "domestic_ligue1":      20.0,
    "domestic_portugal":    20.0,
}


def get_k(competition: str) -> float:
    return K_BY_COMPETITION.get(competition, 20.0)


def run_adaptive_elo():
    df = pd.read_parquet(PROC / "matches_all_multi.parquet")
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Corriendo ELO adaptativo sobre {len(df):,} partidos cronológicos")
    print(f"K por competición:")
    for comp, k in sorted(K_BY_COMPETITION.items(), key=lambda x: -x[1]):
        mult = k / 20.0
        print(f"  {comp:22s}  K={k:.0f}  (x{mult:.1f} vs base)")

    ratings = {}
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        rh = ratings.get(h, DEFAULT_INITIAL)
        ra = ratings.get(a, DEFAULT_INITIAL)
        k = get_k(row["competition"])
        gh, ga = int(row["home_goals"]), int(row["away_goals"])
        if gh > ga: actual = 1.0
        elif gh < ga: actual = 0.0
        else: actual = 0.5
        exp = expected_score(rh, ra, home_adv=DEFAULT_HOME_ADV)
        mult = goal_diff_multiplier(gh - ga)
        delta = k * mult * (actual - exp)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    return ratings


def main():
    ratings = run_adaptive_elo()
    print(f"\nTotal equipos: {len(ratings)}")

    # Guardar
    out = PROC / "final_elo_ratings_adaptive.json"
    with open(out, "w") as f:
        json.dump({t: round(r, 2) for t, r in ratings.items()}, f, indent=2, ensure_ascii=False)
    print(f"Guardado: {out}")

    # Actualizar mini_final_elo.json para predictores
    mini = ROOT / "mini/mini_final_elo.json"
    with open(mini, "w") as f:
        json.dump({t: round(r, 2) for t, r in ratings.items()}, f, indent=2, ensure_ascii=False)

    # Top 25
    print(f"\n🏆 TOP 25 con K ADAPTATIVO (x2.5 UCL / x2.0 UEL / x1.5 UECL):")
    print(f"{'Pos':<4} {'Equipo':<22} {'ELO':>8}  {'ΔvsPreviousK':>14}")
    print("─" * 60)

    # Cargar ELO anterior (K constante) para comparar
    prev_path = PROC / "final_elo_ratings_multi.json"
    if prev_path.exists():
        prev = json.load(open(prev_path))
    else:
        prev = {}

    top = sorted(ratings.items(), key=lambda kv: -kv[1])[:25]
    for i, (t, r) in enumerate(top, 1):
        prev_r = prev.get(t)
        if prev_r is not None:
            delta = r - prev_r
            marker = f"{delta:+7.1f}"
        else:
            marker = "  new"
        print(f"{i:<4} {t:<22} {r:>8.1f}  {marker:>14}")


if __name__ == "__main__":
    main()
