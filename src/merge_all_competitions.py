"""
Unifica matches_all (5 grandes ligas), Champions League, y Portugal Primeira en
un único dataset consolidado ordenado cronológicamente para re-entrenar el ELO
con cross-league calibration.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT = Path("/home/Projects/futbol_elo_poisson")
sys.path.insert(0, str(ROOT / "src"))

# Alias openfootball → nombre canonical (el que usamos en Football-Data)
CHAMPIONS_ALIAS = {
    # España
    "Real Madrid": "Real Madrid", "Real Madrid CF": "Real Madrid",
    "Barcelona": "Barcelona", "FC Barcelona": "Barcelona",
    "Atlético de Madrid": "Ath Madrid", "Club Atlético de Madrid": "Ath Madrid",
    "Atlético Madrid": "Ath Madrid",
    "Athletic Bilbao": "Ath Bilbao", "Athletic Club": "Ath Bilbao", "Athletic Club de Bilbao": "Ath Bilbao",
    "Real Sociedad": "Sociedad",
    "Real Betis": "Betis", "Real Betis Balompié": "Betis",
    "Sevilla": "Sevilla", "Sevilla FC": "Sevilla",
    "Valencia": "Valencia", "Valencia CF": "Valencia",
    "Villarreal": "Villarreal", "Villarreal CF": "Villarreal",
    "Celta": "Celta", "Celta de Vigo": "Celta", "RC Celta de Vigo": "Celta",
    "Girona FC": "Girona", "Girona": "Girona",
    "Getafe": "Getafe", "Getafe CF": "Getafe",
    "Málaga": "Malaga", "Málaga CF": "Malaga",
    # Alemania
    "FC Bayern München": "Bayern Munich", "Bayern München": "Bayern Munich",
    "Bayern Munich": "Bayern Munich", "FC Bayern Munich": "Bayern Munich",
    "Bor. Mönchengladbach": "M'gladbach",
    "Borussia Dortmund": "Dortmund", "BV Borussia 09 Dortmund": "Dortmund", "Dortmund": "Dortmund",
    "Bayer 04 Leverkusen": "Leverkusen", "Bayer Leverkusen": "Leverkusen",
    "RB Leipzig": "RB Leipzig", "RasenBallsport Leipzig": "RB Leipzig",
    "Borussia Mönchengladbach": "M'gladbach", "VfL Borussia Mönchengladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "VfB Stuttgart": "Stuttgart",
    "TSG 1899 Hoffenheim": "Hoffenheim", "1899 Hoffenheim": "Hoffenheim",
    "FC Schalke 04": "Schalke 04",
    "Hertha BSC": "Hertha",
    "1. FC Union Berlin": "Union Berlin",
    # Inglaterra
    "Liverpool FC": "Liverpool", "Liverpool": "Liverpool",
    "Manchester City FC": "Man City", "Manchester City": "Man City",
    "Manchester United FC": "Man United", "Manchester United": "Man United",
    "Chelsea FC": "Chelsea", "Chelsea": "Chelsea",
    "Arsenal FC": "Arsenal", "Arsenal": "Arsenal",
    "Tottenham Hotspur FC": "Tottenham", "Tottenham Hotspur": "Tottenham",
    "Aston Villa FC": "Aston Villa", "Aston Villa": "Aston Villa",
    "Newcastle United FC": "Newcastle", "Newcastle United": "Newcastle",
    "Leicester City FC": "Leicester", "Leicester City": "Leicester",
    # Italia
    "Juventus FC": "Juventus", "Juventus": "Juventus",
    "FC Internazionale Milano": "Inter", "Internazionale": "Inter", "Inter": "Inter",
    "AC Milan": "Milan", "Milan": "Milan",
    "SSC Napoli": "Napoli", "Napoli": "Napoli",
    "AS Roma": "Roma", "Roma": "Roma",
    "Atalanta BC": "Atalanta", "Atalanta": "Atalanta",
    "SS Lazio": "Lazio", "Lazio": "Lazio",
    "ACF Fiorentina": "Fiorentina",
    "Bologna FC 1909": "Bologna",
    # Francia
    "Paris Saint-Germain FC": "Paris SG", "Paris Saint-Germain": "Paris SG", "PSG": "Paris SG",
    "AS Monaco FC": "Monaco", "AS Monaco": "Monaco",
    "Olympique Lyonnais": "Lyon", "OL": "Lyon",
    "Olympique de Marseille": "Marseille", "Marseille": "Marseille",
    "LOSC Lille": "Lille", "Lille OSC": "Lille",
    "OGC Nice": "Nice",
    "AS Saint-Étienne": "St Etienne",
    "Stade Rennais FC": "Rennes",
    "Stade Brestois 29": "Brest",
    "RC Lens": "Lens",
    "Toulouse FC": "Toulouse",
    # Portugal
    "SL Benfica": "Benfica", "Sport Lisboa e Benfica": "Benfica",
    "FC Porto": "Porto", "Futebol Clube do Porto": "Porto",
    "Sporting Clube de Portugal": "Sp Lisbon", "Sporting CP": "Sp Lisbon",
    "SC Braga": "Braga",
    # Otros europeos (mantener nombres originales — no están en nuestras 5 grandes)
}


LEAGUE_MAP_TO_FBREF = {
    "premier": "ENG-Premier League",
    "laliga": "ESP-La Liga",
    "seriea": "ITA-Serie A",
    "bundesliga": "GER-Bundesliga",
    "ligue1": "FRA-Ligue 1",
    "portugal": "POR-Primeira Liga",
}


def load_domestic():
    df = pd.read_parquet(ROOT / "data/processed/matches_all.parquet")
    df["competition"] = "domestic_" + df["league"]
    return df


def load_champions():
    df = pd.read_parquet(ROOT / "data/raw/champions/champions_2015-2026.parquet")
    # Normalizar nombres via alias
    df["home_team_norm"] = df["home_team"].apply(lambda t: CHAMPIONS_ALIAS.get(t, t))
    df["away_team_norm"] = df["away_team"].apply(lambda t: CHAMPIONS_ALIAS.get(t, t))
    # Formato compatible con matches_all
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "league": "champions",
        "season": df["season"].apply(lambda s: f"20{s[:2]}-{s[3:5]}"),
        "home_team": df["home_team_norm"],
        "away_team": df["away_team_norm"],
        "home_goals": df["home_goals"].astype("Int64"),
        "away_goals": df["away_goals"].astype("Int64"),
        "result": df.apply(lambda r: "H" if r["home_goals"] > r["away_goals"]
                          else ("A" if r["home_goals"] < r["away_goals"] else "D"), axis=1),
        "pinnacle_home": float("nan"),
        "pinnacle_draw": float("nan"),
        "pinnacle_away": float("nan"),
        "competition": "champions_league",
    })
    return out


def main():
    print(">>> Merge multi-competición")
    dom = load_domestic()
    print(f"Domésticos: {len(dom):,} partidos")

    champs = load_champions()
    print(f"Champions:  {len(champs):,} partidos")

    # Filtrar champions solo desde 2017-08 (que es cuando arranca nuestro dataset)
    champs_filtered = champs[champs["date"] >= dom["date"].min()].copy()
    print(f"Champions filtrados (post {dom['date'].min().date()}): {len(champs_filtered):,}")

    # Aviso: partidos Champions con equipos NO conocidos en nuestro dataset se mantendrán como equipos nuevos
    known_teams = set(dom["home_team"]) | set(dom["away_team"])
    champ_teams = set(champs_filtered["home_team"]) | set(champs_filtered["away_team"])
    new_teams = champ_teams - known_teams
    print(f"Equipos Champions NO conocidos: {len(new_teams)} (empezarán ELO 1500)")
    print(f"  Muestra: {sorted(list(new_teams))[:20]}")

    # Unificar
    combined = pd.concat([dom, champs_filtered], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    combined = combined.dropna(subset=["home_goals", "away_goals"])

    print(f"\nTotal consolidado: {len(combined):,} partidos")
    print(f"Rango: {combined['date'].min().date()} → {combined['date'].max().date()}")
    print(f"Distribución por competición:")
    print(combined["competition"].value_counts().to_string())

    out = ROOT / "data/processed/matches_all_multi.parquet"
    combined.to_parquet(out)
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()
