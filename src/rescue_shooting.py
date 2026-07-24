"""
Rescate: los datos shooting están en cache (620 archivos HTML).
soccerdata los parsea pero pandas → pyarrow falla por tipo mezclado en col 'GF'.
Solución: forzar tipos consistentes ANTES de guardar.
"""
from pathlib import Path
import pandas as pd
import soccerdata as sd

CACHE = Path("/home/Projects/futbol_elo_poisson/data/raw/fbref_cache")
OUT = Path("/home/Projects/futbol_elo_poisson/data/raw/fbref_light")
OUT.mkdir(exist_ok=True)

LEAGUES = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]
SEASONS = ["17-18", "18-19", "19-20", "20-21", "21-22", "22-23", "23-24", "24-25"]

print(">>> Rescate shooting (usa cache local, no scrape)")
fbref = sd.FBref(leagues=LEAGUES, seasons=SEASONS, data_dir=CACHE)
df = fbref.read_team_match_stats(stat_type="shooting", opponent_stats=False)
print(f"Cargado: {df.shape}, cols: {df.columns.tolist()}")

# Reset index (multiindex → columnas normales)
df = df.reset_index()
print(f"Tras reset_index: {df.shape}, cols: {df.columns.tolist()[:15]}...")

# Aplanar columnas multi-level si existen
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ["_".join(str(c) for c in col if c).strip("_") for col in df.columns]
    print(f"Cols aplanadas: {df.columns.tolist()[:15]}...")

# Forzar tipos numéricos donde debería ser numérico
for col in df.columns:
    if df[col].dtype == object:
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass

# Convertir a CSV para inspección más robusta que parquet ante tipos raros
out_csv = OUT / "team_match_shooting.csv"
df.to_csv(out_csv, index=False)
print(f"CSV guardado: {out_csv} ({out_csv.stat().st_size/1e6:.1f} MB)")

# Y también parquet forzando strings donde falle
try:
    df.to_parquet(OUT / "team_match_shooting.parquet")
    print("Parquet guardado OK")
except Exception as e:
    print(f"Parquet falló: {e}. CSV es suficiente para siguiente paso.")

# Detectar columnas xG
xg_cols = [c for c in df.columns if 'xg' in c.lower() or 'xG' in c]
print(f"\nColumnas xG detectadas: {xg_cols}")
print(f"\nMuestra 3 filas primeras columnas:")
print(df.head(3).to_string())
