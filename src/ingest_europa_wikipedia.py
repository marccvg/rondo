"""
Scraper Wikipedia para UEFA Europa League histórica.
Bajar 9 temporadas 2017-2026.

Estrategia:
- Para cada temporada, intentar bajar 3 páginas potenciales:
  - main season page
  - league_phase (formato Swiss 24/25+)
  - group_stage (formato clásico antes)
  - knockout_phase
- Extraer todas las tablas de resultados
- Consolidar
"""
from __future__ import annotations

import logging
import re
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("/home/Projects/futbol_elo_poisson")
OUT = ROOT / "data/raw/europa"
OUT.mkdir(exist_ok=True, parents=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; educational research; Python)"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("europa")

SEASONS = [
    (2017, 2018), (2018, 2019), (2019, 2020), (2020, 2021),
    (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025), (2025, 2026),
]


def fetch(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.text
    except Exception as e:
        log.warning(f"fetch fail {url}: {e}")
    return None


def wiki_urls_for_season(y1: int, y2: int) -> list[str]:
    """Devuelve URLs candidatas Wikipedia (páginas donde suele haber resultados)."""
    y1s = str(y1); y2s = f"{y2 % 100:02d}"
    slug = f"{y1}%E2%80%93{y2s}"
    return [
        f"https://en.wikipedia.org/wiki/{slug}_UEFA_Europa_League_league_phase",
        f"https://en.wikipedia.org/wiki/{slug}_UEFA_Europa_League_group_stage",
        f"https://en.wikipedia.org/wiki/{slug}_UEFA_Europa_League_knockout_phase",
    ]


# Patrón de score: "1–2", "0–3", "2–2 (a.e.t.)", "2 (5)–2 (3) (a.e.t.)"
SCORE_PATTERN = re.compile(r'^\s*(\d+)\s*[–\-]\s*(\d+)')


def parse_score(s: str) -> tuple[int, int] | None:
    if pd.isna(s):
        return None
    m = SCORE_PATTERN.match(str(s))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def extract_matches_from_tables(html: str, season_label: str) -> list[dict]:
    """Busca tablas con formato ['Home', 'Score', 'Away'] o similar."""
    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        log.warning(f"read_html fail: {e}")
        return []

    partidos = []
    for t in tables:
        # Aplanar multi-index columns
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [" ".join(str(c).strip() for c in col if c and str(c) != "nan").strip() for col in t.columns]
        cols = [str(c).lower() for c in t.columns]
        # Detectar formato "Home / Score / Away"
        home_col = next((c for c in t.columns if str(c).lower() in ["home", "home team"]), None)
        away_col = next((c for c in t.columns if str(c).lower() in ["away", "away team"]), None)
        score_col = next((c for c in t.columns if str(c).lower() in ["score", "result"]), None)
        if home_col is None or away_col is None or score_col is None:
            continue
        for _, row in t.iterrows():
            score = parse_score(row[score_col])
            if score is None:
                continue
            home = str(row[home_col]).strip()
            away = str(row[away_col]).strip()
            if not home or not away or home == "nan" or away == "nan":
                continue
            # Limpiar sufijos tipo " (H)" o coeficientes
            home = re.sub(r'\s*\([^)]*\)\s*$', '', home).strip()
            away = re.sub(r'\s*\([^)]*\)\s*$', '', away).strip()
            partidos.append({
                "home_team": home,
                "away_team": away,
                "home_goals": score[0],
                "away_goals": score[1],
                "season": season_label,
            })
    return partidos


def main():
    all_matches = []
    for y1, y2 in SEASONS:
        season_label = f"{y1 - 2000:02d}-{(y2) - 2000:02d}"
        log.info(f"=== Temporada {season_label} ===")
        seen = set()
        urls = wiki_urls_for_season(y1, y2)
        for url in urls:
            html = fetch(url)
            if not html:
                log.info(f"   [skip] {url.split('/')[-1]}: HTTP no OK")
                continue
            partidos = extract_matches_from_tables(html, season_label)
            # Deduplicar dentro de temporada (misma tupla home-away-scores)
            new = 0
            for p in partidos:
                key = (p["home_team"], p["away_team"], p["home_goals"], p["away_goals"], season_label)
                if key not in seen:
                    seen.add(key)
                    all_matches.append(p)
                    new += 1
            log.info(f"   {url.split('/')[-1]}: {new} partidos nuevos")
            time.sleep(1)  # respect rate limits

    df = pd.DataFrame(all_matches)
    log.info(f"\nTotal Europa League: {len(df):,} partidos")
    if len(df):
        log.info(f"Por temporada:\n{df.groupby('season').size().to_string()}")
        out = OUT / "europa_league_wikipedia.parquet"
        df.to_parquet(out)
        log.info(f"Guardado: {out}")


if __name__ == "__main__":
    main()
