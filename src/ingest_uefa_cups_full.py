"""
Scraper Wikipedia completo para Europa League y Conference League.
Maneja ambos formatos:
  - Group stage cross-tables (17/18-23/24 EL, 21/22-23/24 CL)
  - League phase Swiss (24/25+ EL/CL)
  - Knockout stage
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
OUT = ROOT / "data/raw/uefa_cups"
OUT.mkdir(exist_ok=True, parents=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (educational research)"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("uefa")

EUROPA_SEASONS = [
    (2017, 2018), (2018, 2019), (2019, 2020), (2020, 2021),
    (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025), (2025, 2026),
]
CONFERENCE_SEASONS = [
    (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025), (2025, 2026),
]

SCORE_PATTERN = re.compile(r'(\d+)\s*[–\-]\s*(\d+)')


def fetch(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        return r.text if r.status_code == 200 else None
    except Exception:
        return None


def parse_score(s) -> tuple[int, int] | None:
    if pd.isna(s) or s in ("", "—", "–", "-"):
        return None
    m = SCORE_PATTERN.search(str(s))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


GARBAGE_TEAMS = {
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "1st leg", "2nd leg", "1st Leg", "2nd Leg",
    "agg.", "agg", "aggregate", "score", "result", "teams",
    "home", "away", "team", "opponent",
    "tie", "unnamed",
}
GARBAGE_KEYWORDS = ["Unnamed:", "Tooltip", "Aggregate score", "2nd leg", "1st leg"]


def clean_team(t: str) -> str:
    """Normaliza y valida nombre de equipo."""
    if pd.isna(t):
        return ""
    t = str(t).strip()
    t = re.sub(r'\[[^\]]+\]', '', t)  # remove [1], [note]
    t = re.sub(r'\s*\([^)]{1,4}\)\s*$', '', t)  # remove (H), (Q)
    t = re.sub(r'^\d+\.\s*', '', t)  # remove "1. "
    t = t.strip()
    if not t or len(t) < 2:
        return ""
    tl = t.lower()
    if tl in GARBAGE_TEAMS:
        return ""
    # KEYWORDS match case-sensitive (protege equipos con "note" o similar en nombre legítimo)
    if any(kw in t for kw in GARBAGE_KEYWORDS):
        return ""
    # Rechazar puramente numérico o con puntuación excesiva
    if re.match(r'^[\d\s\.\-–,;:]+$', t):
        return ""
    return t


def extract_from_cross_tables(html: str, season_label: str) -> list[dict]:
    """Extrae resultados de tablas cruzadas de group stage clásico.
    Detecta: si columnas (después de header) contienen nombres de equipos
    que también aparecen en la columna Team, es formato cruzado.
    """
    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        return []

    partidos = []
    for t in tables:
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [" ".join(str(c).strip() for c in col if c and str(c) != "nan").strip()
                         for col in t.columns]

        cols = list(t.columns)
        if len(cols) < 3:
            continue

        # Detectar cross-table:
        # 1. Primera columna suele ser "Team" o similar
        # 2. Resto de columnas suelen ser nombres de equipos con score en cells
        team_col_idx = None
        for i, c in enumerate(cols[:2]):
            cell_sample = str(t.iloc[0, i]) if len(t) > 0 else ""
            if not SCORE_PATTERN.search(cell_sample):  # celda sin score
                team_col_idx = i
                break
        if team_col_idx is None:
            continue

        # Otras columnas: deben contener nombres de equipos
        other_col_idxs = [j for j in range(len(cols)) if j != team_col_idx]
        # Verificar que al menos una celda tiene formato score
        has_score_cells = False
        for _, row in t.iterrows():
            for j in other_col_idxs:
                if parse_score(row.iloc[j]):
                    has_score_cells = True
                    break
            if has_score_cells:
                break
        if not has_score_cells:
            continue

        # Es cross-table: extraer resultados
        team_names_in_col = [clean_team(str(v)) for v in t.iloc[:, team_col_idx]]
        # Los headers de columnas (excluyendo team_col) también deberían ser nombres de equipos
        # o abreviaciones. Wikipedia a veces usa abreviaciones en headers.
        for i, row in t.iterrows():
            home = clean_team(str(row.iloc[team_col_idx]))
            if not home:
                continue
            for j in other_col_idxs:
                col_name = cols[j]
                # El "away" es el nombre de la columna
                away = clean_team(str(col_name))
                if not away or away == home:
                    continue
                score = parse_score(row.iloc[j])
                if score is None:
                    continue
                partidos.append({
                    "home_team": home,
                    "away_team": away,
                    "home_goals": score[0],
                    "away_goals": score[1],
                    "season": season_label,
                    "phase": "group_cross",
                })
    return partidos


def extract_from_linear_tables(html: str, season_label: str) -> list[dict]:
    """Extrae de tablas linear ['Home', 'Score', 'Away'] (Swiss + knockout)."""
    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        return []

    partidos = []
    for t in tables:
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [" ".join(str(c).strip() for c in col if c and str(c) != "nan").strip()
                         for col in t.columns]
        home_col = next((c for c in t.columns if str(c).lower() in ["home", "home team"]), None)
        away_col = next((c for c in t.columns if str(c).lower() in ["away", "away team"]), None)
        score_col = next((c for c in t.columns if str(c).lower() in ["score", "result"]), None)
        if not (home_col and away_col and score_col):
            continue
        for _, row in t.iterrows():
            score = parse_score(row[score_col])
            if score is None:
                continue
            home = clean_team(str(row[home_col]))
            away = clean_team(str(row[away_col]))
            if not home or not away:
                continue
            partidos.append({
                "home_team": home,
                "away_team": away,
                "home_goals": score[0],
                "away_goals": score[1],
                "season": season_label,
                "phase": "linear",
            })
    return partidos


def urls_for(competition: str, y1: int, y2: int) -> list[str]:
    slug = f"{y1}%E2%80%93{y2 % 100:02d}"
    if competition == "europa":
        base = "UEFA_Europa_League"
    elif competition == "conference":
        base = "UEFA_Europa_Conference_League" if y1 < 2024 else "UEFA_Conference_League"
    else:
        raise ValueError(competition)

    return [
        f"https://en.wikipedia.org/wiki/{slug}_{base}_league_phase",     # 24/25+
        f"https://en.wikipedia.org/wiki/{slug}_{base}_group_stage",     # <=23/24
        f"https://en.wikipedia.org/wiki/{slug}_{base}_knockout_phase",
        f"https://en.wikipedia.org/wiki/{slug}_{base}_knockout_stage",
    ]


def scrape_competition(competition: str, seasons: list[tuple[int, int]]) -> pd.DataFrame:
    all_matches = []
    for y1, y2 in seasons:
        season_label = f"{y1 - 2000:02d}-{y2 - 2000:02d}"
        log.info(f"=== {competition} {season_label} ===")
        seen = set()
        for url in urls_for(competition, y1, y2):
            html = fetch(url)
            if not html:
                continue
            # Probar ambos parsers
            partidos = extract_from_linear_tables(html, season_label)
            partidos += extract_from_cross_tables(html, season_label)
            new = 0
            for p in partidos:
                key = (p["home_team"], p["away_team"], p["home_goals"], p["away_goals"], season_label)
                if key not in seen:
                    seen.add(key)
                    p["competition"] = competition
                    all_matches.append(p)
                    new += 1
            log.info(f"   {url.split('/')[-1][:60]}: {new} nuevos")
            time.sleep(1)
    return pd.DataFrame(all_matches)


def main():
    europa_df = scrape_competition("europa", EUROPA_SEASONS)
    conf_df = scrape_competition("conference", CONFERENCE_SEASONS)

    log.info(f"\nEuropa League: {len(europa_df)} partidos")
    if len(europa_df):
        log.info(f"Por temporada:\n{europa_df.groupby('season').size().to_string()}")
        europa_df.to_parquet(OUT / "europa_league_full.parquet")

    log.info(f"\nConference League: {len(conf_df)} partidos")
    if len(conf_df):
        log.info(f"Por temporada:\n{conf_df.groupby('season').size().to_string()}")
        conf_df.to_parquet(OUT / "conference_league_full.parquet")


if __name__ == "__main__":
    main()
