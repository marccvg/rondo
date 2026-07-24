"""
Auto-actualizador del proyecto.

Cada ejecución:
1. Detecta qué temporadas nuevas faltan en local vs las disponibles públicamente.
2. Descarga incremental de Football-Data.co.uk (CSVs) y opcionalmente Understat (xG).
3. Reconstruye matches_all.parquet consolidado.
4. Recalcula ELO histórico completo.
5. Guarda snapshot de ratings.
6. Reporta cambios importantes vs snapshot anterior.

Uso:
    venv/bin/python pipeline/refresh.py                        # actualiza data + ELO
    venv/bin/python pipeline/refresh.py --with-understat       # incluye xG (más lento)
    venv/bin/python pipeline/refresh.py --predict laliga       # actualiza + predice
    venv/bin/python pipeline/refresh.py --predict premier
    venv/bin/python pipeline/refresh.py --predict champions
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

ROOT = Path("/home/Projects/futbol_elo_poisson")
sys.path.insert(0, str(ROOT / "src"))

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SNAPSHOTS = ROOT / "data" / "snapshots"
SNAPSHOTS.mkdir(exist_ok=True, parents=True)

# Football-Data.co.uk códigos oficiales
FD_LEAGUES = {
    "E0": "Premier League",
    "SP1": "LaLiga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "P1": "Primeira Liga",   # Portugal
}

INITIAL_SEASON_YEAR = 2017  # temporada 17/18 = primera del dataset


def season_code(year: int) -> str:
    """year=2025 → '2526' (temporada 25/26)."""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def get_target_seasons(today: date) -> list[int]:
    """Determina qué temporadas queremos cubrir.
    Regla: temporada N/N+1 termina en mayo N+1. Si hoy es antes de agosto, la temporada N-1/N recién terminó.
    Si estamos en agosto+, la temporada N/N+1 ya empezó.
    Queremos todas desde INITIAL_SEASON_YEAR hasta la última completa o en curso.
    """
    # Temporada más reciente que podría tener datos: la que empezó en agosto del año calendar anterior o actual
    if today.month >= 8:
        latest_year = today.year
    else:
        latest_year = today.year - 1
    return list(range(INITIAL_SEASON_YEAR, latest_year + 1))


def check_url_exists(url: str, timeout: int = 15) -> bool:
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False


def download_footballdata_incremental() -> dict:
    """Descarga los CSVs de Football-Data que faltan localmente."""
    print("\n📥 Fase 1 — descarga incremental Football-Data.co.uk")
    out_dir = RAW / "footballdata"
    out_dir.mkdir(exist_ok=True, parents=True)

    today = date.today()
    target_years = get_target_seasons(today)
    print(f"   Temporadas objetivo: {[f'{y % 100:02d}/{(y+1)%100:02d}' for y in target_years]}")

    downloaded, skipped, failed = [], [], []
    for lg_code, lg_name in FD_LEAGUES.items():
        for year in target_years:
            code = season_code(year)
            fname = out_dir / f"{lg_code}_{code}.csv"
            url = f"https://www.football-data.co.uk/mmz4281/{code}/{lg_code}.csv"

            if fname.exists() and fname.stat().st_size > 1000:
                # Ya existe. Para la temporada MÁS RECIENTE re-descargar (pueden haber añadido jornadas).
                is_current = (year == target_years[-1])
                if not is_current:
                    skipped.append(fname.name)
                    continue

            try:
                r = requests.get(url, timeout=30)
                if r.status_code != 200 or len(r.content) < 1000:
                    print(f"   [skip] {lg_name} {code}: no disponible aún (HTTP {r.status_code}, {len(r.content)}B)")
                    failed.append(fname.name)
                    continue
                fname.write_bytes(r.content)
                action = "actualizado" if fname.exists() else "nuevo"
                print(f"   [ok] {lg_name} {year % 100:02d}/{(year+1)%100:02d} → {fname.name} ({len(r.content):,} B)")
                downloaded.append(fname.name)
                time.sleep(0.5)
            except Exception as e:
                print(f"   [fail] {lg_name} {code}: {e}")
                failed.append(fname.name)

    print(f"   Resumen: {len(downloaded)} descargados, {len(skipped)} ya existían, {len(failed)} no disponibles.")
    return {"downloaded": downloaded, "skipped": skipped, "failed": failed}


def download_understat_incremental() -> dict:
    """Descarga schedule Understat con xG para temporadas nuevas."""
    print("\n📥 Fase 2 — descarga incremental Understat (xG)")
    from pathlib import Path
    import soccerdata as sd

    cache = RAW / "understat_cache"
    out_dir = RAW / "understat_full"
    out_dir.mkdir(exist_ok=True, parents=True)

    today = date.today()
    target_years = get_target_seasons(today)

    LEAGUES = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]

    frames = []
    ok, fail = 0, 0
    for league in LEAGUES:
        for year in target_years:
            season = str(year)  # Understat usa año único
            try:
                us = sd.Understat(leagues=league, seasons=season, data_dir=cache)
                sched = us.read_schedule().reset_index()
                sched["_league_norm"] = league
                sched["_season_year"] = season
                frames.append(sched)
                ok += 1
                print(f"   [ok] {league} {season}: {len(sched)} partidos")
            except Exception as e:
                fail += 1
                print(f"   [skip] {league} {season}: {type(e).__name__}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        if isinstance(combined.columns, pd.MultiIndex):
            combined.columns = ["_".join(str(c) for c in col if c).strip("_") for col in combined.columns]
        out = out_dir / "understat_schedule_5big.parquet"
        combined.to_parquet(out)
        print(f"   Consolidado: {len(combined):,} partidos → {out}")
    return {"ok": ok, "fail": fail}


def rebuild_matches_parquet():
    """Re-consolida matches_all.parquet a partir de los CSVs actuales."""
    print("\n🔧 Fase 3 — reconstruir matches_all.parquet")
    from data_loader import load_all_footballdata
    df = load_all_footballdata()
    out = PROC / "matches_all.parquet"
    df.to_parquet(out)
    print(f"   {len(df):,} partidos, {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"   Guardado: {out}")
    return df


def refresh_champions():
    """Bajar/actualizar Champions League de openfootball."""
    print("\n🏆 Fase 3b — actualizar Champions League (openfootball)")
    import subprocess
    result = subprocess.run(
        [str(ROOT / "venv/bin/python"), str(ROOT / "src/ingest_champions.py")],
        capture_output=True, text=True, timeout=120,
    )
    for line in result.stdout.splitlines()[-6:]:
        print(f"   {line}")


def rebuild_multi_and_elo():
    """Fusiona domestic + Champions, recomputa ELO cross-league."""
    print("\n🔀 Fase 4b — merge multi-competición + ELO cross-league")
    import subprocess
    for script in ["src/merge_all_competitions.py", "src/elo_multi.py"]:
        result = subprocess.run(
            [str(ROOT / "venv/bin/python"), str(ROOT / script)],
            capture_output=True, text=True, timeout=180,
        )
        for line in result.stdout.splitlines()[-4:]:
            print(f"   {line}")
    # Actualizar mini/mini_final_elo.json con el cross-league (usado por predictores)
    import shutil
    src = PROC / "final_elo_ratings_multi.json"
    dst = ROOT / "mini/mini_final_elo.json"
    if src.exists():
        shutil.copy(src, dst)
        print(f"   ELO cross-league copiado a {dst.name} (para predictores)")


def rebuild_elo(df: pd.DataFrame) -> dict:
    """Recomputa ELO histórico y guarda snapshot fresco."""
    print("\n📊 Fase 4 — recomputar ELO")
    from elo import compute_elo_history, EloState, expected_score

    df_elo = compute_elo_history(df)

    # Reconstruir ratings finales (last known)
    state = EloState()
    for _, row in df_elo.iterrows():
        state.update(row["home_team"], row["away_team"], int(row["home_goals"]), int(row["away_goals"]))

    ratings = state.ratings
    print(f"   Corrido sobre {len(df_elo):,} partidos, {len(ratings)} equipos únicos")

    # Guardar
    out_parquet = PROC / "matches_with_elo.parquet"
    df_elo.to_parquet(out_parquet)

    ratings_out = PROC / "final_elo_ratings.json"
    with open(ratings_out, "w") as f:
        json.dump({t: round(r, 2) for t, r in ratings.items()}, f, indent=2, ensure_ascii=False)
    print(f"   Ratings finales → {ratings_out}")

    # También actualizar el file de mini/ para compatibilidad con learning/
    mini_ratings = ROOT / "mini" / "mini_final_elo.json"
    with open(mini_ratings, "w") as f:
        json.dump({t: round(r, 2) for t, r in ratings.items()}, f, indent=2, ensure_ascii=False)

    # Snapshot con timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap = SNAPSHOTS / f"elo_snapshot_{ts}.json"
    with open(snap, "w") as f:
        json.dump({t: round(r, 2) for t, r in ratings.items()}, f, indent=2, ensure_ascii=False)
    print(f"   Snapshot histórico → {snap}")

    return ratings


def compare_with_previous_snapshot(current_ratings: dict) -> None:
    """Compara ratings actuales vs el snapshot inmediatamente anterior."""
    print("\n📈 Fase 5 — comparar vs snapshot anterior")
    snapshots = sorted(SNAPSHOTS.glob("elo_snapshot_*.json"))
    if len(snapshots) < 2:
        print("   No hay snapshot previo para comparar.")
        return

    prev_path = snapshots[-2]
    prev = json.load(open(prev_path))
    print(f"   Comparando con: {prev_path.name}")

    changes = []
    for team, current in current_ratings.items():
        if team in prev:
            delta = current - prev[team]
            if abs(delta) >= 5:
                changes.append((team, prev[team], current, delta))

    changes.sort(key=lambda x: -abs(x[3]))
    if not changes:
        print("   Sin cambios significativos (>5 ELO puntos) desde último snapshot.")
        return

    print(f"   Cambios >5 puntos ELO (top 15):")
    print(f"   {'Equipo':<25} {'Anterior':>10} {'Actual':>10} {'Delta':>10}")
    for team, p, c, d in changes[:15]:
        sign = "↑" if d > 0 else "↓"
        print(f"   {team:<25} {p:>10.1f} {c:>10.1f}   {sign}{abs(d):>6.1f}")

    new_teams = set(current_ratings.keys()) - set(prev.keys())
    if new_teams:
        print(f"\n   ⭐ Equipos nuevos que no estaban antes: {sorted(new_teams)}")


def top_ratings(ratings: dict, n: int = 20) -> None:
    print(f"\n🏆 Top {n} equipos por ELO (post-actualización):")
    top = sorted(ratings.items(), key=lambda kv: -kv[1])[:n]
    for i, (t, r) in enumerate(top, 1):
        print(f"   {i:2d}. {t:25s} {r:7.1f}")


def run_prediction(league_key: str, ratings: dict) -> None:
    """Regenera predicción para la próxima temporada usando los ratings actualizados."""
    print(f"\n🔮 Fase 6 — predicción {league_key}")
    if league_key.lower() == "champions":
        import subprocess
        result = subprocess.run(
            [str(ROOT / "venv/bin/python"), str(ROOT / "learning/predict_champions_next_season.py")],
            capture_output=True, text=True,
        )
        # Filtrar sección clave del output
        lines = result.stdout.split("\n")
        # Print desde donde aparece "CAMPEÓN"
        printing = False
        for line in lines:
            if "CAMPEÓN" in line and "🏆" in line:
                printing = True
            if printing:
                print(f"   {line}")
    elif league_key.lower() in ("laliga", "premier"):
        import subprocess
        result = subprocess.run(
            [str(ROOT / "venv/bin/python"), str(ROOT / "learning/predict_ligas_next_season.py")],
            capture_output=True, text=True,
        )
        # Print output del predictor
        for line in result.stdout.split("\n")[-100:]:
            print(f"   {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-understat", action="store_true",
                    help="Incluir descarga Understat xG (lento, ~5-10 min por Selenium)")
    ap.add_argument("--predict", choices=["laliga", "premier", "champions", "all"],
                    help="Regenerar predicción tras actualizar")
    ap.add_argument("--no-download", action="store_true", help="Solo recomputar ELO sin descargar")
    args = ap.parse_args()

    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print(f"║  REFRESH — {datetime.now().strftime('%Y-%m-%d %H:%M')}                                       ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")

    if not args.no_download:
        download_footballdata_incremental()
        refresh_champions()  # ← NUEVO: Champions League desde openfootball
        if args.with_understat:
            download_understat_incremental()
    else:
        print("Skip descargas (--no-download).")

    df = rebuild_matches_parquet()
    ratings = rebuild_elo(df)                # ELO solo domésticas (guardado como final_elo_ratings.json)
    rebuild_multi_and_elo()                  # ← NUEVO: ELO cross-league con Champions (guardado como final_elo_ratings_multi.json)
    compare_with_previous_snapshot(ratings)
    top_ratings(ratings, n=20)

    if args.predict:
        preds = ["laliga", "premier", "champions"] if args.predict == "all" else [args.predict]
        for p in preds:
            run_prediction(p, ratings)

    print("\n✅ Refresh terminado.\n")


if __name__ == "__main__":
    main()
