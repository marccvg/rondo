"""
ELO con K derivado del coeficiente UEFA por país + K UEFA cups fijo.

K UEFA cups (nivel competición):
  UCL: 50, UEL: 40, UECL: 30

K ligas domésticas (derivado del UEFA country coefficient calculado sobre nuestros datos):
  England:  20.00  (top)
  Spain:    17.52
  Portugal: 15.96
  Germany:  15.43
  Italy:    15.19
  France:   13.13
"""
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from elo import expected_score, goal_diff_multiplier, DEFAULT_INITIAL, DEFAULT_HOME_ADV

ROOT = Path("/home/Projects/futbol_elo_poisson")
PROC = ROOT / "data" / "processed"

# K UEFA cups (mantiene approach anterior)
K_UEFA = {
    "champions_league":  50.0,
    "europa_league":     40.0,
    "conference_league": 30.0,
}

# K por liga derivado UEFA coefficient (cargar del JSON generado)
country_to_k = json.load(open(PROC / "uefa_country_k.json"))

COMP_TO_COUNTRY = {
    "domestic_premier": "England",
    "domestic_laliga": "Spain",
    "domestic_seriea": "Italy",
    "domestic_bundesliga": "Germany",
    "domestic_ligue1": "France",
    "domestic_portugal": "Portugal",
}


def get_k(competition: str) -> float:
    if competition in K_UEFA:
        return K_UEFA[competition]
    country = COMP_TO_COUNTRY.get(competition)
    if country and country in country_to_k:
        return country_to_k[country]
    return 20.0  # default


def main():
    df = pd.read_parquet(PROC / "matches_all_multi.parquet")
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Corriendo ELO CALIBRADO UEFA sobre {len(df):,} partidos cronológicos")
    print(f"\nK por competición:")
    for comp in sorted(set(df["competition"])):
        k = get_k(comp)
        print(f"  {comp:22s}  K={k:.2f}")

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

    print(f"\nTotal equipos: {len(ratings)}")

    # Guardar
    out = PROC / "final_elo_ratings_uefa_calibrated.json"
    with open(out, "w") as f:
        json.dump({t: round(r, 2) for t, r in ratings.items()}, f, indent=2, ensure_ascii=False)
    print(f"Guardado: {out}")

    # Cargar rating anterior para comparar
    prev = json.load(open(PROC / "final_elo_ratings_adaptive.json"))

    print(f"\n🏆 TOP 25 con K UEFA CALIBRADO:")
    print(f"{'Pos':<4} {'Equipo':<22} {'ELO':>8}  {'ΔvsPrevKAdap':>14}")
    print("─" * 62)
    top = sorted(ratings.items(), key=lambda kv: -kv[1])[:25]
    for i, (t, r) in enumerate(top, 1):
        pr = prev.get(t)
        d = f"{r-pr:+7.1f}" if pr is not None else "  new"
        print(f"{i:<4} {t:<22} {r:>8.1f}  {d:>14}")

    # Copiar a mini para predictores
    with open(ROOT / "mini/mini_final_elo.json", "w") as f:
        json.dump({t: round(r, 2) for t, r in ratings.items()}, f, indent=2, ensure_ascii=False)
    # También a multi para predictores que lo cargan
    with open(PROC / "final_elo_ratings_multi.json", "w") as f:
        json.dump({t: round(r, 2) for t, r in ratings.items()}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
