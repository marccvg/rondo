"""
Ingesta FBref completa — 5 grandes ligas + Champions × 8 temporadas.
Metricas por partido y por equipo: schedule, shooting, passing, defense, possession, misc.
"""
from __future__ import annotations

import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import soccerdata as sd

ROOT = Path("/home/Projects/futbol_elo_poisson")
RAW = ROOT / "data" / "raw"
CACHE = RAW / "fbref_cache"
OUT = RAW / "fbref_advanced"
OUT.mkdir(exist_ok=True)

LOG_PATH = RAW / "fbref_ingest.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fbref_ingest")

LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]
# Champions League NO está en el league_dict built-in de soccerdata.
# Se puede añadir via ~/soccerdata/config/league_dict.json custom si se quiere en el futuro.

SEASONS = ["17-18", "18-19", "19-20", "20-21", "21-22", "22-23", "23-24", "24-25"]

STAT_TYPES_TEAM = [
    "schedule",
    "shooting",
    "passing",
    "passing_types",
    "goal_shot_creation",
    "defense",
    "possession",
    "misc",
    "keeper",
]


def main():
    log.info(">>> FBref ingesta completa iniciada <<<")
    log.info(f"Ligas: {LEAGUES}")
    log.info(f"Temporadas: {SEASONS}")
    log.info(f"Stat types: {STAT_TYPES_TEAM}")

    # Instanciar UN objeto con todo — soccerdata gestiona rate limit internamente
    try:
        fbref = sd.FBref(leagues=LEAGUES, seasons=SEASONS, data_dir=CACHE)
        log.info("Instancia FBref creada OK")
    except Exception as e:
        log.error(f"[FATAL] no pude instanciar FBref: {e}")
        log.error(traceback.format_exc())
        return

    for stat_type in STAT_TYPES_TEAM:
        out_path = OUT / f"team_match_{stat_type}.parquet"
        if out_path.exists():
            log.info(f"[skip] {stat_type} ya existe en {out_path.name}")
            continue
        log.info(f"=== read_team_match_stats(stat_type={stat_type!r}) ===")
        t0 = time.time()
        try:
            df = fbref.read_team_match_stats(stat_type=stat_type, opponent_stats=True)
            elapsed = time.time() - t0
            log.info(f"[ok  ] {stat_type}: {len(df):,} filas, {len(df.columns)} cols, {elapsed:.1f}s")
            df.to_parquet(out_path)
        except Exception as e:
            elapsed = time.time() - t0
            log.warning(f"[fail] {stat_type} tras {elapsed:.1f}s: {type(e).__name__}: {e}")
            # No abortamos — seguimos con los demás stat_types
            continue

    log.info(">>> FBref ingesta completa terminada <<<")
    # Resumen
    for stat_type in STAT_TYPES_TEAM:
        p = OUT / f"team_match_{stat_type}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            log.info(f"  ✓ {stat_type}: {len(df):,} filas, {len(df.columns)} cols")
        else:
            log.info(f"  ✗ {stat_type}: NO descargado")


if __name__ == "__main__":
    main()
