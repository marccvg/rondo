"""
Bajar player_match_stats de LaLiga vía FBref (soccerdata).

Datos que baja por jugador y partido:
- Goles, asistencias, minutos
- xG, xA
- Pases progresivos, tiros, tackles
- Tarjetas, penaltis

Cobertura: LaLiga 2017-18 hasta 2025-26 = 9 temporadas
Volumen esperado: ~50.000-70.000 rows (jugador-partido)
Tiempo estimado: 1-2h (scraping vía Selenium con rate limits FBref)
"""
from __future__ import annotations
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import soccerdata as sd

ROOT = Path("/home/Projects/futbol_elo_poisson/fantasy_laliga")
RAW = ROOT / "data" / "raw"
RAW.mkdir(exist_ok=True, parents=True)

CACHE = Path("/home/Projects/futbol_elo_poisson/data/raw/fbref_cache")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[
        logging.FileHandler(RAW / "laliga_ingest.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("laliga_ingest")

SEASONS = ["17-18", "18-19", "19-20", "20-21", "21-22", "22-23", "23-24", "24-25", "25-26"]


def main():
    log.info(">>> Ingesta LaLiga player stats — 9 temporadas")
    all_dfs = []

    for season in SEASONS:
        log.info(f"→ LaLiga {season}...")
        try:
            fbref = sd.FBref(leagues="ESP-La Liga", seasons=season, data_dir=CACHE)
            t0 = time.time()
            # Bajamos player_match_stats con varios stat types
            for stat_type in ["summary", "keeper", "passing", "defense", "misc"]:
                try:
                    df = fbref.read_player_match_stats(stat_type=stat_type)
                    df["_stat_type"] = stat_type
                    df["_season"] = season
                    all_dfs.append(df.reset_index())
                    log.info(f"   {stat_type}: {len(df):,} rows")
                except Exception as e:
                    log.warning(f"   {stat_type}: fail {type(e).__name__}: {e}")
                time.sleep(1)
            elapsed = time.time() - t0
            log.info(f"   → {season} completado en {elapsed:.0f}s")
        except Exception as e:
            log.error(f"  fail {season}: {e}")
            continue

    if not all_dfs:
        log.error("Nada descargado")
        return

    log.info("Consolidando...")
    combined = pd.concat(all_dfs, ignore_index=True, sort=False)
    out = RAW / "laliga_player_stats_all.parquet"
    combined.to_parquet(out)
    log.info(f"✅ Guardado {len(combined):,} rows → {out}")


if __name__ == "__main__":
    main()
