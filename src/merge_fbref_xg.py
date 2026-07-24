"""
Merge Football-Data.co.uk matches con FBref schedule para añadir xG por partido.

Reto: nombres de equipos difieren entre fuentes ("Manchester Utd" vs "Man United").
Estrategia: fuzzy match sobre (fecha, liga, home, away).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("/home/Projects/futbol_elo_poisson")
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"


# Mapeo FootballData → FBref (los que difieren)
TEAM_ALIAS = {
    "Man United": "Manchester Utd",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nott'ham Forest",
    "Ath Madrid": "Atlético Madrid",
    "Ath Bilbao": "Athletic Club",
    "Betis": "Betis",
    "Vallecano": "Rayo Vallecano",
    "Alaves": "Alavés",
    "Cadiz": "Cádiz",
    "Almeria": "Almería",
    "Malaga": "Málaga",
    "La Coruna": "La Coruña",
    "Leganes": "Leganés",
    "Espanol": "Espanyol",
    "Ein Frankfurt": "Eint Frankfurt",
    "M'gladbach": "Gladbach",
    "FC Koln": "Köln",
    "Nurnberg": "Nürnberg",
    "Fortuna Dusseldorf": "Düsseldorf",
    "Hertha": "Hertha BSC",
    "Leverkusen": "Leverkusen",
    "Paris SG": "Paris S-G",
    "St Etienne": "Saint-Étienne",
    "Nimes": "Nîmes",
    "Marseille": "Marseille",
    "Verona": "Hellas Verona",
    "Sassuolo": "Sassuolo",
    "Inter": "Internazionale",
}

LEAGUE_TO_FBREF = {
    "premier": "ENG-Premier League",
    "laliga": "ESP-La Liga",
    "seriea": "ITA-Serie A",
    "bundesliga": "GER-Bundesliga",
    "ligue1": "FRA-Ligue 1",
}


def apply_alias(team: str) -> str:
    return TEAM_ALIAS.get(team, team)


def main():
    matches = pd.read_parquet(PROC / "matches_all.parquet")
    print(f"FootballData matches: {len(matches):,}")

    fbref_path = RAW / "fbref_light" / "schedule_5big_17-25.parquet"
    if not fbref_path.exists():
        print(f"FALTA {fbref_path} — ingesta FBref light aún no ha terminado")
        return
    fbref = pd.read_parquet(fbref_path)
    print(f"FBref schedule: {len(fbref):,} filas, cols: {fbref.columns.tolist()[:20]}...")

    # Detectar columnas xG
    xg_home_col = next((c for c in fbref.columns if 'xg' in c.lower() and 'home' in c.lower()), None)
    xg_away_col = next((c for c in fbref.columns if 'xg' in c.lower() and 'away' in c.lower()), None)
    if not xg_home_col:
        # fbref suele usar 'xG' (mayúscula) o similar
        print(f"WARN: no encuentro xG_home. Cols: {[c for c in fbref.columns if 'xG' in c or 'xg' in c.lower()]}")

    print(f"xG cols detectadas: home={xg_home_col}, away={xg_away_col}")

    # Normalizar equipos
    matches["_home_norm"] = matches["home_team"].apply(apply_alias)
    matches["_away_norm"] = matches["away_team"].apply(apply_alias)
    matches["_league_norm"] = matches["league"].map(LEAGUE_TO_FBREF)
    matches["_date_norm"] = pd.to_datetime(matches["date"]).dt.date

    # Preparar fbref
    fbref["_home_norm"] = fbref.get("home_team", "")
    fbref["_away_norm"] = fbref.get("away_team", "")
    fbref["_date_norm"] = pd.to_datetime(fbref.get("date", pd.NaT)).dt.date

    # Merge por (date, league, home, away)
    keep_cols = ["_home_norm", "_away_norm", "_date_norm", "_league_norm"]
    if xg_home_col:
        keep_cols.append(xg_home_col)
    if xg_away_col:
        keep_cols.append(xg_away_col)
    fbref_slim = fbref[[c for c in keep_cols if c in fbref.columns]].copy()

    merged = matches.merge(
        fbref_slim,
        on=["_date_norm", "_league_norm", "_home_norm", "_away_norm"],
        how="left",
        suffixes=("", "_fbref"),
    )
    print(f"Match rate: {(merged[xg_home_col].notna().sum() / len(merged) * 100):.1f}% partidos con xG" if xg_home_col else "N/A")

    # Renombrar a xg_home / xg_away
    if xg_home_col:
        merged = merged.rename(columns={xg_home_col: "xg_home", xg_away_col: "xg_away"})

    out = PROC / "matches_with_xg.parquet"
    merged.drop(columns=[c for c in merged.columns if c.startswith("_")]).to_parquet(out)
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
