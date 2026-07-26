"""
Ingesta histórico completo Fantasy Premier League.
Fuente: github.com/vaastav/Fantasy-Premier-League (comunidad, actualizado semanalmente).
Formato: merged_gw.csv por temporada con TODOS los jugadores y jornadas.
"""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import requests

ROOT = Path("/home/Projects/futbol_elo_poisson/fantasy")
OUT = ROOT / "data" / "raw"
OUT.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("fpl_hist")

# Temporadas disponibles en el repo
SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
           "2022-23", "2023-24", "2024-25", "2025-26"]


def download_season(season):
    """Baja merged_gw.csv de una temporada."""
    url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"
    log.info(f"→ {season}: {url}")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            log.warning(f"  fail {r.status_code}")
            return None
        # Guardar CSV raw
        (OUT / f"fpl_merged_{season}.csv").write_bytes(r.content)
        df = pd.read_csv(pd.io.common.BytesIO(r.content))
        df["_season"] = season
        log.info(f"  ✅ {len(df):,} rows, cols: {len(df.columns)}")
        return df
    except Exception as e:
        log.warning(f"  fail {e}")
        return None


def main():
    all_dfs = []
    for season in SEASONS:
        df = download_season(season)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        log.error("Nada descargado")
        return

    # Consolidar
    combined = pd.concat(all_dfs, ignore_index=True, sort=False)
    log.info(f"\n📊 TOTAL consolidado: {len(combined):,} rows")
    log.info(f"Rango temporadas: {sorted(combined['_season'].unique())}")
    log.info(f"Cols: {combined.columns.tolist()}")

    out = OUT / "fpl_history_all_seasons.parquet"
    combined.to_parquet(out)
    log.info(f"Guardado: {out}")

    # Stats rápidos
    log.info(f"\nJugadores únicos: {combined['name'].nunique() if 'name' in combined.columns else 'N/A'}")
    log.info(f"Sample: \n{combined.head(3).to_string()[:500]}")


if __name__ == "__main__":
    main()
