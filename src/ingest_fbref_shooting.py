"""
Ingesta FBref específica para stat_type='shooting' — trae xG por partido.

Aprovecha el cache parcial (~205 archivos) del intento pesado abortado.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import soccerdata as sd

ROOT = Path("/home/Projects/futbol_elo_poisson")
RAW = ROOT / "data" / "raw"
CACHE = RAW / "fbref_cache"
OUT = RAW / "fbref_light"
OUT.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[
        logging.FileHandler(RAW / "fbref_shooting.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("fbref_shooting")

LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]
SEASONS = ["17-18", "18-19", "19-20", "20-21", "21-22", "22-23", "23-24", "24-25"]


def main():
    log.info(">>> FBref shooting ingesta iniciada (xG por partido) <<<")
    t_global = time.time()
    fbref = sd.FBref(leagues=LEAGUES, seasons=SEASONS, data_dir=CACHE)
    log.info("Instancia FBref OK, llamando read_team_match_stats(shooting)...")
    try:
        df = fbref.read_team_match_stats(stat_type="shooting", opponent_stats=False)
        elapsed = time.time() - t_global
        log.info(f"[OK] {len(df):,} filas, {len(df.columns)} cols, {elapsed:.1f}s")
        out = OUT / "team_match_shooting.parquet"
        df.to_parquet(out)
        log.info(f"Guardado: {out}")
        # Detectar xG cols
        xg_cols = [c for c in df.columns if 'xg' in c.lower()]
        log.info(f"Cols xG: {xg_cols}")
        log.info(f"Todas las cols: {df.columns.tolist()}")
    except Exception as e:
        log.error(f"[FAIL] {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
