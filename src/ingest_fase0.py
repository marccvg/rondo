"""
Fase 0 — ingesta datos.

Estrategia:
1. Fuentes CSV/git (rápidas, fiables): Football-Data.co.uk, StatsBomb open, ClubELO.
2. Fuentes scraping (frágiles): FBref, Understat — probar con muestra pequeña primero.

Todo va a data/raw/. Log a data/raw/ingest.log. Al final: data/INVENTORY.md.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("/home/Projects/futbol_elo_poisson")
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

LOG_PATH = RAW / "ingest.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ingest")

# Ligas objetivo Football-Data.co.uk (códigos oficiales)
FD_LEAGUES = {
    "E0": "Premier League",
    "SP1": "LaLiga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}
# Temporadas: 2017/18 .. 2024/25. Código FD: "1718", "1819", ...
FD_SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2017, 2025)]


def step_footballdata() -> dict:
    """Descarga CSVs de Football-Data.co.uk. Rapido, fiable."""
    log.info("=== step_footballdata ===")
    out_dir = RAW / "footballdata"
    out_dir.mkdir(exist_ok=True)
    stats = {"ok": 0, "fail": 0, "files": []}
    for lg_code, lg_name in FD_LEAGUES.items():
        for season in FD_SEASONS:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{lg_code}.csv"
            fname = out_dir / f"{lg_code}_{season}.csv"
            if fname.exists() and fname.stat().st_size > 1000:
                log.info(f"[skip] {fname.name} ya existe ({fname.stat().st_size} bytes)")
                stats["ok"] += 1
                continue
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                fname.write_bytes(r.content)
                log.info(f"[ok  ] {lg_name} {season} → {fname.name} ({len(r.content)} bytes)")
                stats["ok"] += 1
                stats["files"].append(fname.name)
                time.sleep(0.5)
            except Exception as e:
                log.warning(f"[fail] {lg_name} {season}: {e}")
                stats["fail"] += 1
    return stats


def step_clubelo() -> dict:
    """ClubELO API — ratings actuales de todos los equipos europeos."""
    log.info("=== step_clubelo ===")
    stats = {"ok": 0, "fail": 0}
    try:
        # Endpoint que devuelve rating de todos los equipos a fecha concreta
        # Formato: http://api.clubelo.com/YYYY-MM-DD
        url = "http://api.clubelo.com/2025-07-24"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        fname = RAW / "clubelo_snapshot_2025-07-24.csv"
        fname.write_bytes(r.content)
        log.info(f"[ok  ] ClubELO snapshot → {fname.name} ({len(r.content)} bytes)")
        stats["ok"] += 1

        # Histórico por club de las top 30 (bajar uno a uno sería lento; para v1 basta snapshot)
        # Pendiente Fase 1 si necesitamos histórico Elo externo
    except Exception as e:
        log.warning(f"[fail] ClubELO: {e}")
        stats["fail"] += 1
    return stats


def step_statsbomb_open() -> dict:
    """Git clone StatsBomb open-data. ~2 GB, event-level."""
    log.info("=== step_statsbomb_open ===")
    target = RAW / "statsbomb_open"
    stats = {"ok": 0, "fail": 0}
    if target.exists():
        log.info(f"[skip] {target} ya existe")
        stats["ok"] += 1
        return stats
    try:
        cmd = ["git", "clone", "--depth", "1", "https://github.com/statsbomb/open-data.git", str(target)]
        log.info(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode == 0:
            size_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e6
            log.info(f"[ok  ] StatsBomb open-data clonado ({size_mb:.0f} MB)")
            stats["ok"] += 1
        else:
            log.warning(f"[fail] git clone stderr: {result.stderr[:500]}")
            stats["fail"] += 1
    except Exception as e:
        log.warning(f"[fail] StatsBomb: {e}")
        stats["fail"] += 1
    return stats


def step_fbref_probe() -> dict:
    """
    FBref via soccerdata — probar con 1 liga 1 temporada para validar que scraping funciona.
    Si OK, en step separado (o job aparte) haremos las 8 temporadas × 6 competiciones.
    """
    log.info("=== step_fbref_probe ===")
    stats = {"ok": 0, "fail": 0, "probe_data": None}
    try:
        import soccerdata as sd
        # Cache dir dentro del proyecto para no contaminar $HOME
        cache_dir = RAW / "fbref_cache"
        cache_dir.mkdir(exist_ok=True)
        fbref = sd.FBref(leagues="ENG-Premier League", seasons="24-25", data_dir=cache_dir)
        log.info("FBref instanciado, probando read_schedule()...")
        sched = fbref.read_schedule()
        log.info(f"[ok  ] FBref probe: {len(sched)} partidos en Premier 24/25")
        # Guardar como parquet para inventario
        out = RAW / "fbref_probe_premier_24-25.parquet"
        sched.to_parquet(out)
        stats["ok"] += 1
        stats["probe_data"] = f"Premier 24/25: {len(sched)} rows, columns={list(sched.columns)[:15]}..."
    except Exception as e:
        log.warning(f"[fail] FBref probe: {type(e).__name__}: {e}")
        log.warning(traceback.format_exc())
        stats["fail"] += 1
    return stats


def step_understat_probe() -> dict:
    """Understat via soccerdata — probar 1 liga 1 temporada."""
    log.info("=== step_understat_probe ===")
    stats = {"ok": 0, "fail": 0, "probe_data": None}
    try:
        import soccerdata as sd
        cache_dir = RAW / "understat_cache"
        cache_dir.mkdir(exist_ok=True)
        us = sd.Understat(leagues="ENG-Premier League", seasons="2024", data_dir=cache_dir)
        sched = us.read_schedule()
        log.info(f"[ok  ] Understat probe: {len(sched)} partidos")
        out = RAW / "understat_probe_premier_2024.parquet"
        sched.to_parquet(out)
        stats["ok"] += 1
        stats["probe_data"] = f"Premier 2024: {len(sched)} rows"
    except Exception as e:
        log.warning(f"[fail] Understat probe: {type(e).__name__}: {e}")
        stats["fail"] += 1
    return stats


def build_inventory(results: dict) -> None:
    """Genera data/INVENTORY.md con lo descargado."""
    log.info("=== build_inventory ===")
    lines = ["# Inventario datos — Fase 0", ""]
    lines.append(f"Generado: {pd.Timestamp.now().isoformat()}")
    lines.append("")
    lines.append("## Resumen por fuente")
    lines.append("")
    lines.append("| Fuente | Descargas OK | Fallos | Estado |")
    lines.append("|---|---|---|---|")
    for src, s in results.items():
        estado = "OK" if s.get("fail", 0) == 0 else ("PARCIAL" if s.get("ok", 0) > 0 else "FALLO")
        lines.append(f"| {src} | {s.get('ok', 0)} | {s.get('fail', 0)} | {estado} |")
    lines.append("")

    # Detalle Football-Data
    fd_dir = RAW / "footballdata"
    if fd_dir.exists():
        lines.append("## Football-Data.co.uk — detalle")
        lines.append("")
        for csv in sorted(fd_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv, encoding="latin-1", on_bad_lines="skip", low_memory=False)
                lines.append(f"- `{csv.name}` — {len(df)} filas, {len(df.columns)} cols")
            except Exception as e:
                lines.append(f"- `{csv.name}` — ERROR leer: {e}")
        lines.append("")

    # Detalle FBref probe
    probe = RAW / "fbref_probe_premier_24-25.parquet"
    if probe.exists():
        df = pd.read_parquet(probe)
        lines.append("## FBref probe (Premier 24/25)")
        lines.append("")
        lines.append(f"- {len(df)} partidos, {len(df.columns)} columnas")
        lines.append(f"- Columnas: {', '.join(df.columns.tolist())}")
        lines.append("")

    # Detalle StatsBomb
    sb = RAW / "statsbomb_open"
    if sb.exists():
        sb_size = sum(f.stat().st_size for f in sb.rglob("*") if f.is_file()) / 1e6
        matches_dir = sb / "data" / "matches"
        n_comps = len(list(matches_dir.glob("*.json"))) if matches_dir.exists() else 0
        lines.append("## StatsBomb Open Data")
        lines.append("")
        lines.append(f"- {sb_size:.0f} MB")
        lines.append(f"- Competiciones (archivos matches/*.json): {n_comps}")
        lines.append("")

    inv_path = ROOT / "data" / "INVENTORY.md"
    inv_path.write_text("\n".join(lines))
    log.info(f"[ok  ] INVENTORY escrito → {inv_path}")


def main():
    log.info(">>> INGESTA FASE 0 iniciada <<<")
    results = {}
    results["footballdata"] = step_footballdata()
    results["clubelo"] = step_clubelo()
    results["statsbomb_open"] = step_statsbomb_open()
    results["fbref_probe"] = step_fbref_probe()
    results["understat_probe"] = step_understat_probe()
    build_inventory(results)
    log.info(">>> INGESTA FASE 0 terminada <<<")
    log.info(f"Resumen: {results}")


if __name__ == "__main__":
    main()
