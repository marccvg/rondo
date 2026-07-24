"""Ingesta Understat completa — schedule con xG_home/xG_away por partido."""
from __future__ import annotations
import logging
import sys
import time
from pathlib import Path
import pandas as pd
import soccerdata as sd

ROOT = Path("/home/Projects/futbol_elo_poisson")
RAW = ROOT / "data" / "raw"
CACHE = RAW / "understat_cache"
OUT = RAW / "understat_full"
OUT.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[
        logging.FileHandler(RAW / "understat_ingest.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("understat")

LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]
# Understat usa año único (2017 = 17/18)
SEASONS = ["2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]


def main():
    log.info(">>> Understat ingesta iniciada <<<")
    frames = []
    for league in LEAGUES:
        for season in SEASONS:
            log.info(f"→ {league} {season}")
            t0 = time.time()
            try:
                us = sd.Understat(leagues=league, seasons=season, data_dir=CACHE)
                sched = us.read_schedule().reset_index()
                sched["_league_norm"] = league
                sched["_season_year"] = season
                frames.append(sched)
                log.info(f"  [ok] {len(sched)} partidos en {time.time()-t0:.1f}s")
            except Exception as e:
                log.warning(f"  [fail] {type(e).__name__}: {e}")

    if not frames:
        log.error("Nada descargado")
        return

    combined = pd.concat(frames, ignore_index=True)
    # Aplanar posibles multi-index
    if isinstance(combined.columns, pd.MultiIndex):
        combined.columns = ["_".join(str(c) for c in col if c).strip("_") for col in combined.columns]
    out = OUT / "understat_schedule_5big.parquet"
    combined.to_parquet(out)
    log.info(f">>> TOTAL: {len(combined):,} partidos, cols: {combined.columns.tolist()}")
    log.info(f"Guardado: {out}")
    log.info(f"Con xG home: {combined['home_xg'].notna().sum():,} ({combined['home_xg'].notna().mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
