"""
Ingesta FBref ligera — read_schedule por liga-temporada.

Trae por partido: home, away, score, xG_home, xG_away, attendance, referee.
40 requests (5 ligas × 8 temp) vs 7,200 del approach heavy.
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
        logging.FileHandler(RAW / "fbref_light.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("fbref_light")

LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]
SEASONS = ["17-18", "18-19", "19-20", "20-21", "21-22", "22-23", "23-24", "24-25"]


def main():
    log.info(">>> FBref LIGHT ingesta iniciada <<<")
    all_schedules = []
    for league in LEAGUES:
        for season in SEASONS:
            log.info(f"→ {league} {season}")
            t0 = time.time()
            try:
                fbref = sd.FBref(leagues=league, seasons=season, data_dir=CACHE)
                sched = fbref.read_schedule()
                sched["_league_norm"] = league
                sched["_season_norm"] = season
                all_schedules.append(sched.reset_index())
                elapsed = time.time() - t0
                log.info(f"  [ok] {len(sched)} partidos en {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.time() - t0
                log.warning(f"  [fail] {type(e).__name__}: {e} tras {elapsed:.1f}s")

    if not all_schedules:
        log.error("Nada descargado")
        return

    combined = pd.concat(all_schedules, ignore_index=True)
    out = OUT / "schedule_5big_17-25.parquet"
    combined.to_parquet(out)
    log.info(f">>> Total: {len(combined):,} partidos, {len(combined.columns)} cols → {out}")
    log.info(f"Columnas: {combined.columns.tolist()}")
    # Check si tenemos xG
    xg_cols = [c for c in combined.columns if 'xg' in c.lower() or 'xG' in c]
    log.info(f"Columnas xG detectadas: {xg_cols}")


if __name__ == "__main__":
    main()
