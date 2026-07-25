"""
Calcula coeficiente UEFA por país aplicando fórmula oficial sobre nuestros datos.

Fórmula UEFA (simplificada, sin bonus rondas por parsimonía):
- Victoria en Champions/Europa/Conference: +2 puntos país
- Empate: +1 punto
- Total puntos país / número equipos participantes = coef temporada
- Coef 5-year = suma últimas 5 temporadas

Salida: coef por país por temporada + coef 5-year rolling.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path("/home/Projects/futbol_elo_poisson")

# Mapping domestic competition -> country name
COUNTRY_BY_COMP = {
    "domestic_premier": "England",
    "domestic_laliga": "Spain",
    "domestic_seriea": "Italy",
    "domestic_bundesliga": "Germany",
    "domestic_ligue1": "France",
    "domestic_portugal": "Portugal",
}


def build_team_to_country(df: pd.DataFrame) -> dict:
    """Asocia cada equipo a su país basado en la liga doméstica donde aparece."""
    team_country = {}
    for comp, country in COUNTRY_BY_COMP.items():
        sub = df[df["competition"] == comp]
        for t in set(sub["home_team"]) | set(sub["away_team"]):
            if t not in team_country:  # first country wins
                team_country[t] = country
    return team_country


def uefa_coef_per_season(df: pd.DataFrame) -> pd.DataFrame:
    """Coef por país por temporada (partidos UEFA cups)."""
    team_country = build_team_to_country(df)

    uefa = df[df["competition"].isin(["champions_league", "europa_league", "conference_league"])].copy()

    country_pts = defaultdict(lambda: defaultdict(float))       # {season: {country: pts}}
    country_teams = defaultdict(lambda: defaultdict(set))       # {season: {country: teams}}

    for _, row in uefa.iterrows():
        season = row["season"]
        hc = team_country.get(row["home_team"])
        ac = team_country.get(row["away_team"])
        if not hc or not ac: continue  # equipos de países no incluidos

        # Sumar equipos participantes
        country_teams[season][hc].add(row["home_team"])
        country_teams[season][ac].add(row["away_team"])

        # Puntos por resultado
        if row["home_goals"] > row["away_goals"]:
            country_pts[season][hc] += 2
        elif row["home_goals"] < row["away_goals"]:
            country_pts[season][ac] += 2
        else:
            country_pts[season][hc] += 1
            country_pts[season][ac] += 1

    # Construir tabla: coef por (temporada, país)
    rows = []
    for season, cp in country_pts.items():
        for country, pts in cp.items():
            n_teams = len(country_teams[season][country])
            coef = pts / n_teams if n_teams > 0 else 0
            rows.append({"season": season, "country": country, "pts_total": pts,
                         "n_teams": n_teams, "coef": round(coef, 3)})
    return pd.DataFrame(rows).sort_values(["season", "country"])


def coef_5year_rolling(coef_df: pd.DataFrame) -> dict:
    """Coef 5-year rolling por país (usa últimas 5 temporadas)."""
    seasons_sorted = sorted(coef_df["season"].unique())
    last5 = seasons_sorted[-5:]
    result = {}
    for country in coef_df["country"].unique():
        c = coef_df[(coef_df["country"] == country) & (coef_df["season"].isin(last5))]
        result[country] = round(c["coef"].sum(), 3)
    return result


def main():
    df = pd.read_parquet(ROOT / "data/processed/matches_all_multi.parquet")
    coef_df = uefa_coef_per_season(df)

    print("═" * 78)
    print("  COEFICIENTE UEFA POR PAÍS Y TEMPORADA (fórmula oficial sobre nuestros datos)")
    print("═" * 78)

    pivot = coef_df.pivot(index="season", columns="country", values="coef").fillna(0)
    # Ordenar columnas por coef 5-year descendente
    coef_5y = coef_5year_rolling(coef_df)
    cols_ordered = sorted(pivot.columns, key=lambda c: -coef_5y.get(c, 0))
    pivot = pivot[cols_ordered]
    print(pivot.to_string())

    print("\n" + "═" * 78)
    print("  COEFICIENTE UEFA 5-YEAR ROLLING (últimas 5 temporadas 21-26)")
    print("═" * 78)
    ranked = sorted(coef_5y.items(), key=lambda x: -x[1])
    max_c = ranked[0][1]
    print(f"\n{'Pos':<4} {'País':<15} {'Coef 5-year':>12}  {'K liga':>8}  Ratio vs top")
    print("─" * 62)
    for i, (country, c) in enumerate(ranked, 1):
        k = round(20 * (c / max_c), 2)
        print(f"{i:<4} {country:<15} {c:>12.3f}  {k:>8.2f}  {c/max_c*100:>10.1f}%")

    # Guardar K por país para usar en ELO adaptativo
    country_to_k = {country: round(20 * (c / max_c), 2) for country, c in coef_5y.items()}
    with open(ROOT / "data/processed/uefa_country_k.json", "w") as f:
        json.dump(country_to_k, f, indent=2)
    print(f"\nGuardado en: data/processed/uefa_country_k.json")


if __name__ == "__main__":
    main()
