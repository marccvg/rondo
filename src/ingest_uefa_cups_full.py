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


def extract_from_ko_tables(html: str, season_label: str) -> list[dict]:
    """Extrae partidos KO stage con formato Team 1 / Team 2 / 1st leg / 2nd leg.
    - 1st leg: Team 1 = local, Team 2 = visitante
    - 2nd leg: Team 2 = local, Team 1 = visitante
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
        cols_str = " ".join(str(c) for c in t.columns).lower()
        # Detectar formato KO
        if not ("team 1" in cols_str and "team 2" in cols_str
                and "1st leg" in cols_str and "2nd leg" in cols_str):
            continue
        # Encontrar columnas exactas
        c_team1 = next((c for c in t.columns if "team 1" in str(c).lower()), None)
        c_team2 = next((c for c in t.columns if "team 2" in str(c).lower()), None)
        c_leg1 = next((c for c in t.columns if "1st leg" in str(c).lower()), None)
        c_leg2 = next((c for c in t.columns if "2nd leg" in str(c).lower()), None)
        if not all([c_team1, c_team2, c_leg1, c_leg2]):
            continue
        for _, row in t.iterrows():
            t1 = clean_team(str(row[c_team1]))
            t2 = clean_team(str(row[c_team2]))
            if not t1 or not t2:
                continue
            # 1st leg: T1 local, T2 visitante
            score1 = parse_score(row[c_leg1])
            if score1:
                partidos.append({
                    "home_team": t1, "away_team": t2,
                    "home_goals": score1[0], "away_goals": score1[1],
                    "season": season_label, "phase": "ko_leg1",
                })
            # 2nd leg: T2 local, T1 visitante
            score2 = parse_score(row[c_leg2])
            if score2:
                partidos.append({
                    "home_team": t2, "away_team": t1,
                    "home_goals": score2[0], "away_goals": score2[1],
                    "season": season_label, "phase": "ko_leg2",
                })
    return partidos


def extract_from_group_standings(html: str, season_label: str) -> list[dict]:
    """Extrae de standings de group stage clásico.
    Wikipedia usa formato: [Pos, Team, Pld, W, D, L, GF, GA, GD, Pts, Qualification, ...códigos rival 3-letras...]
    Las últimas N columnas son códigos de 3 letras del rival, celdas contienen 'X–Y' resultado.
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
        cols = [str(c) for c in t.columns]
        # Detectar: columnas típicas standings + N columnas al final que parecen códigos 3-letras
        standings_cols = {"Pos", "Team", "Teamvte", "Pld", "W", "D", "L", "GF", "GA", "GD", "Pts"}
        matched_std = sum(1 for c in cols if c in standings_cols)
        if matched_std < 5:
            continue
        # Encontrar columna Team
        team_col = next((c for c in cols if "team" in c.lower()), None)
        if not team_col:
            continue
        # Encontrar columnas de códigos: al final, 3-4 letras mayúsculas
        code_cols = [c for c in cols if len(c) == 3 and c.isupper() and c.isalpha()]
        if len(code_cols) < 3:
            continue
        # Mapear: código 3-letras → nombre completo
        # Los códigos suelen ser primeras 3 letras del nombre en mayúscula
        teams_in_group = [clean_team(str(v)) for v in t[team_col] if clean_team(str(v))]
        code_to_full = {}
        for full in teams_in_group:
            # Generar candidatos código posibles
            candidates = [full[:3].upper()]
            words = full.split()
            if len(words) > 1:
                candidates.append("".join(w[0] for w in words[:3]).upper())
            for c in candidates:
                if c in code_cols and c not in code_to_full:
                    code_to_full[c] = full
                    break
        # Si no matcheamos los códigos por naming, matchear por posición (mismo orden filas ~ columnas)
        if len(code_to_full) < len(code_cols):
            # Assumir que orden columnas código = orden equipos en tabla
            code_to_full = {code_cols[i]: teams_in_group[i] for i in range(min(len(code_cols), len(teams_in_group)))}

        # Extraer partidos
        for _, row in t.iterrows():
            home = clean_team(str(row[team_col]))
            if not home:
                continue
            for code in code_cols:
                away = code_to_full.get(code)
                if not away or away == home:
                    continue
                score = parse_score(row[code])
                if score is None:
                    continue
                partidos.append({
                    "home_team": home,
                    "away_team": away,
                    "home_goals": score[0],
                    "away_goals": score[1],
                    "season": season_label,
                    "phase": "group_stage",
                })
    return partidos


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

    urls = [
        # Fase principal
        f"https://en.wikipedia.org/wiki/{slug}_{base}_league_phase",     # 24/25+
        f"https://en.wikipedia.org/wiki/{slug}_{base}_group_stage",     # <=23/24
        # Knockout
        f"https://en.wikipedia.org/wiki/{slug}_{base}_knockout_phase",
        f"https://en.wikipedia.org/wiki/{slug}_{base}_knockout_stage",
        f"https://en.wikipedia.org/wiki/{slug}_{base}_knockout_round_play-offs",
    ]
    # Qualifying rounds - Europa vs Conference usan URLs distintas
    if competition == "europa":
        urls += [
            f"https://en.wikipedia.org/wiki/{slug}_{base}_qualifying_rounds_and_play-off_round",
            f"https://en.wikipedia.org/wiki/{slug}_{base}_first_qualifying_round",
            f"https://en.wikipedia.org/wiki/{slug}_{base}_second_qualifying_round",
            f"https://en.wikipedia.org/wiki/{slug}_{base}_third_qualifying_round",
            f"https://en.wikipedia.org/wiki/{slug}_{base}_play-off_round",
        ]
    elif competition == "conference":
        # Conference usa URLs con paréntesis
        urls += [
            f"https://en.wikipedia.org/wiki/{slug}_{base}_qualifying_(first_and_second_round_matches)",
            f"https://en.wikipedia.org/wiki/{slug}_{base}_qualifying_(third_and_play-off_round_matches)",
            f"https://en.wikipedia.org/wiki/{slug}_{base}_qualifying",
        ]
    return urls


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
            # Probar todos los parsers
            partidos = extract_from_linear_tables(html, season_label)
            partidos += extract_from_group_standings(html, season_label)
            partidos += extract_from_cross_tables(html, season_label)
            partidos += extract_from_ko_tables(html, season_label)
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
