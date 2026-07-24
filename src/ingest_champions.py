"""
Ingesta Champions League histórica desde openfootball.
Formato source: texto plano ("Team A (COUNTRY) v Team B (COUNTRY) 3-1 (2-0)").
Output: data/raw/champions/champions_2015-2026.parquet
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("/home/Projects/futbol_elo_poisson")
OUT = ROOT / "data" / "raw" / "champions"
OUT.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("champions")

# Regex para líneas con fecha
DATE_LINE = re.compile(r'^\s+(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\w+)\s+(\d+)(?:\s+(\d{4}))?')
# Regex para líneas con partido: "  [HH:MM] Home Team (XXX) v Away Team (YYY) X-Y (X-Y)"
MATCH_LINE = re.compile(
    r'^\s+(?:(\d{2}):(\d{2})\s+)?'                # opcional hora
    r'(.+?)\s+\(([A-Z]{2,3})\)\s+'                # home team + país
    r'v\s+'
    r'(.+?)\s+\(([A-Z]{2,3})\)\s+'                # away team + país
    r'(\d+)-(\d+)'                                # score final
    r'(?:\s+\((\d+)-(\d+)\))?'                    # opcional HT score
)

MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def parse_season(txt: str, season_str: str) -> list[dict]:
    """Parsea el contenido del cl.txt para una temporada.
    season_str: '2024-25'. Se usa para inferir año en fechas sin año explícito."""
    year_start = int("20" + season_str[:2])  # 2024
    year_end = year_start + 1

    current_date = None
    current_month = None
    current_day = None
    current_year = year_start
    stage = None
    partidos = []

    for line in txt.splitlines():
        line_stripped = line.rstrip()
        if not line_stripped:
            continue

        # Detectar stage: "▪ League, Matchday 1", "▪ Playoffs", etc.
        if line_stripped.startswith("▪"):
            stage = line_stripped.lstrip("▪ ").strip()
            continue

        # Detectar fecha
        m_date = DATE_LINE.match(line)
        if m_date:
            dow, month, day, year = m_date.groups()
            current_month = MONTHS.get(month)
            current_day = int(day)
            # Inferir año: partidos entre Ago-Dic → año_inicio; Ene-Jul → año_fin
            if year:
                current_year = int(year)
            else:
                current_year = year_start if current_month and current_month >= 7 else year_end
            try:
                current_date = pd.Timestamp(current_year, current_month, current_day).date()
            except Exception:
                current_date = None
            continue

        # Detectar partido
        m_match = MATCH_LINE.match(line)
        if m_match and current_date:
            hh, mm, home, home_c, away, away_c, gh, ga, hth, hta = m_match.groups()
            partidos.append({
                "date": current_date,
                "home_team": home.strip(),
                "home_country": home_c,
                "away_team": away.strip(),
                "away_country": away_c,
                "home_goals": int(gh),
                "away_goals": int(ga),
                "ht_home": int(hth) if hth else None,
                "ht_away": int(hta) if hta else None,
                "stage": stage,
                "season": season_str,
                "competition": "champions_league",
            })
    return partidos


def download_and_parse_season(season: str) -> list[dict]:
    year_start = int("20" + season[:2])
    year_end = int(season[3:5])  # '25' from '2024-25'
    url = f"https://raw.githubusercontent.com/openfootball/champions-league/master/{year_start}-{year_end:02d}/cl.txt"
    log.info(f"Downloading {season}: {url}")
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        log.warning(f"  fail {r.status_code}")
        return []
    partidos = parse_season(r.text, season)
    log.info(f"  parseados: {len(partidos)} partidos")
    return partidos


def main():
    SEASONS = [f"{y - 2000:02d}-{(y + 1) - 2000:02d}" for y in range(2015, 2026)]
    log.info(f"Objetivo: {len(SEASONS)} temporadas Champions")

    all_partidos = []
    for season in SEASONS:
        all_partidos.extend(download_and_parse_season(season))

    df = pd.DataFrame(all_partidos)
    df["date"] = pd.to_datetime(df["date"])
    log.info(f"\nTotal: {len(df):,} partidos Champions")
    log.info(f"Rango fechas: {df['date'].min()} → {df['date'].max()}")
    log.info(f"Por temporada:\n{df.groupby('season').size()}")

    out = OUT / "champions_2015-2026.parquet"
    df.to_parquet(out)
    log.info(f"Guardado: {out}")

    # Sample
    log.info(f"\nMuestra:\n{df.head(3).to_string()}")


if __name__ == "__main__":
    main()
