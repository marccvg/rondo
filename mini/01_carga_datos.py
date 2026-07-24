"""
MINI paso 1 — cargar solo los 4 datos que usaremos y ver qué son.
"""
import pandas as pd
from pathlib import Path

df = pd.read_parquet("/home/Projects/futbol_elo_poisson/data/processed/matches_all.parquet")

# Nos quedamos SOLO con lo mínimo
mini = df[["date", "home_team", "away_team", "home_goals", "away_goals"]].copy()
mini["home_goals"] = mini["home_goals"].astype(int)
mini["away_goals"] = mini["away_goals"].astype(int)
mini["result"] = mini.apply(
    lambda r: "H" if r["home_goals"] > r["away_goals"]
              else ("A" if r["home_goals"] < r["away_goals"] else "D"),
    axis=1,
)

print("=" * 60)
print("  MINI paso 1 — DATOS DISPONIBLES")
print("=" * 60)
print(f"\nTotal partidos: {len(mini):,}")
print(f"Rango fechas: {mini['date'].min()} → {mini['date'].max()}")
print(f"Equipos únicos: {len(set(mini['home_team']) | set(mini['away_team']))}")

print("\nPrimeras 3 filas:")
print(mini.head(3).to_string(index=False))

print("\nÚltimas 3 filas:")
print(mini.tail(3).to_string(index=False))

print("\nDistribución resultado:")
print(mini["result"].value_counts(normalize=True).round(3).to_string())

print("\nDistribución goles home (top-8):")
print(mini["home_goals"].value_counts().sort_index().head(8).to_string())

print("\nDistribución goles away (top-8):")
print(mini["away_goals"].value_counts().sort_index().head(8).to_string())

# Guardar mini_matches para siguientes scripts
out = Path("/home/Projects/futbol_elo_poisson/mini/mini_matches.parquet")
mini.to_parquet(out)
print(f"\nGuardado en {out}")
