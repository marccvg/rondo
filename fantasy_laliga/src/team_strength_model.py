"""
Modelo baseline LaLiga Fantasy — predicción de fuerza ofensiva/defensiva por EQUIPO.

Con esto podemos ranking equipos → deduces qué jugadores rendirán mejor:
  - Delanteros de equipos con alto xG generado (Barça, Real Madrid) → más goles esperados
  - Defensas de equipos con bajo xG concedido (equipos sólidos atrás) → más clean sheets
  - Mediocampistas de equipos balanceados → asistencias + puntos por defensa

Es baseline utilizable HASTA que llegue el scraping player_match_stats
(1-2h en background) donde tendremos modelo por jugador individual.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path("/home/Projects/futbol_elo_poisson")
OUT = ROOT / "fantasy_laliga/data/processed"
OUT.mkdir(exist_ok=True, parents=True)


def compute_team_strengths():
    df = pd.read_parquet(ROOT / "data/processed/matches_all_multi.parquet")
    elos = json.load(open(ROOT / "data/processed/final_elo_ratings_uefa_calibrated.json"))

    # Filtrar LaLiga (últimas 3 temporadas para reflejar plantilla actual)
    df = df[
        (df["competition"] == "domestic_laliga") &
        (df["season"].isin(["2023-24", "2024-25", "2025-26"]))
    ].copy()
    print(f"Base: {len(df):,} partidos LaLiga últimas 3 temporadas")

    # Merge con xG Understat
    try:
        understat = pd.read_parquet(ROOT / "data/raw/understat_full/understat_schedule_5big.parquet")
        understat = understat[understat.get("_league_norm", "") == "ESP-La Liga"]
        # No aplico merge estricto por complejidad — uso goles reales que ya están en df
    except Exception:
        pass

    # Stats por equipo (agregadas 3 temporadas)
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))

    strengths = []
    for team in teams:
        home_matches = df[df["home_team"] == team]
        away_matches = df[df["away_team"] == team]

        gf = home_matches["home_goals"].sum() + away_matches["away_goals"].sum()
        ga = home_matches["away_goals"].sum() + away_matches["home_goals"].sum()
        n_matches = len(home_matches) + len(away_matches)
        clean_sheets = (home_matches["away_goals"] == 0).sum() + (away_matches["home_goals"] == 0).sum()

        if n_matches == 0:
            continue

        strengths.append({
            "team": team,
            "elo": elos.get(team, 1500),
            "matches": n_matches,
            "goals_per_match": gf / n_matches,
            "goals_conceded_per_match": ga / n_matches,
            "clean_sheet_ratio": clean_sheets / n_matches,
            # Ratings de utilidad fantasy
            "attacking_strength": (gf / n_matches) / 1.3,  # normalizado ~media LaLiga 1.3 gpp
            "defensive_strength": 1 - min(1, (ga / n_matches) / 1.3),  # 0-1
        })

    df_str = pd.DataFrame(strengths).sort_values("elo", ascending=False)
    print(f"\n📊 Fuerza equipos LaLiga (últimas 3 temp):")
    print(df_str.round(3).to_string(index=False))

    # Ranking utilidad fantasy por rol
    print("\n\n🎯 TOP 8 ATTACKING teams (buenos DELANTEROS aquí):")
    print(df_str.nlargest(8, "goals_per_match")[["team", "elo", "goals_per_match"]].to_string(index=False))

    print("\n🛡️ TOP 8 DEFENSIVE teams (buenos DEFENSAS/PORTEROS aquí):")
    print(df_str.nlargest(8, "clean_sheet_ratio")[["team", "elo", "goals_conceded_per_match", "clean_sheet_ratio"]].to_string(index=False))

    # Guardar
    df_str.to_csv(OUT / "team_strengths.csv", index=False)
    print(f"\nGuardado: {OUT / 'team_strengths.csv'}")

    return df_str


def predict_team_fantasy_output(df_str, home_team, away_team):
    """Dado enfrentamiento, predice output fantasy esperado para cada equipo."""
    h = df_str[df_str["team"] == home_team]
    a = df_str[df_str["team"] == away_team]
    if h.empty or a.empty:
        print(f"⚠️ Faltan datos para {home_team} o {away_team}")
        return None
    h, a = h.iloc[0], a.iloc[0]

    print(f"\n{'═'*60}")
    print(f"  {home_team}  vs  {away_team}")
    print(f"{'═'*60}")
    print(f"\nELOs: {home_team} {h['elo']:.0f} — {away_team} {a['elo']:.0f}")
    print(f"ELO diff: {h['elo']-a['elo']:+.0f}")

    # Estimación puntos fantasy esperados
    expected_h_goals = h["goals_per_match"] * (1 - a["defensive_strength"] * 0.5)
    expected_a_goals = a["goals_per_match"] * (1 - h["defensive_strength"] * 0.5)

    print(f"\nGoles esperados:")
    print(f"  {home_team}: {expected_h_goals:.2f}  (local, favorecido)")
    print(f"  {away_team}: {expected_a_goals:.2f}")

    p_clean_sheet_h = max(0, 1 - expected_a_goals * 0.6)
    p_clean_sheet_a = max(0, 1 - expected_h_goals * 0.6)

    print(f"\nProb clean sheet:")
    print(f"  {home_team}: {p_clean_sheet_h*100:.1f}%")
    print(f"  {away_team}: {p_clean_sheet_a*100:.1f}%")

    # Puntos fantasy esperados por posición-equipo
    from scoring_system import POINTS_GOAL, POINTS_CLEAN_SHEET

    print(f"\n📈 Puntos fantasy esperados por 90 min (aproximado, por jugador titular):")
    print(f"\n  {home_team} (local):")
    print(f"    Portero:  {2 + p_clean_sheet_h*POINTS_CLEAN_SHEET['GK'] - expected_a_goals*1:.2f}")
    print(f"    Defensa:  {2 + p_clean_sheet_h*POINTS_CLEAN_SHEET['DEF']:.2f}")
    print(f"    Delantero:{2 + expected_h_goals*POINTS_GOAL['FWD']/3.5:.2f}  (dividido entre ~3.5 delanteros/mediapuntas)")

    print(f"\n  {away_team} (visitante):")
    print(f"    Portero:  {2 + p_clean_sheet_a*POINTS_CLEAN_SHEET['GK'] - expected_h_goals*1:.2f}")
    print(f"    Defensa:  {2 + p_clean_sheet_a*POINTS_CLEAN_SHEET['DEF']:.2f}")
    print(f"    Delantero:{2 + expected_a_goals*POINTS_GOAL['FWD']/3.5:.2f}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    df_str = compute_team_strengths()

    # Ejemplo de uso
    predict_team_fantasy_output(df_str, "Real Madrid", "Barcelona")
    predict_team_fantasy_output(df_str, "Barcelona", "Betis")
    predict_team_fantasy_output(df_str, "Ath Madrid", "Villarreal")
