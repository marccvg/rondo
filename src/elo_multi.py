"""
Recomputa ELO sobre el dataset multi-competición (domésticos + Champions).
Guarda ratings para comparar con la versión sin Champions.
"""
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path("/home/Projects/futbol_elo_poisson")
sys.path.insert(0, str(ROOT / "src"))

from elo import EloState

df = pd.read_parquet(ROOT / "data/processed/matches_all_multi.parquet")
df = df.sort_values("date").reset_index(drop=True)
print(f"Corriendo ELO sobre {len(df):,} partidos ({df['date'].min().date()} → {df['date'].max().date()})")

state = EloState()
for _, row in df.iterrows():
    state.update(row["home_team"], row["away_team"], int(row["home_goals"]), int(row["away_goals"]))

ratings = state.ratings
print(f"Total equipos únicos: {len(ratings)}")

out = ROOT / "data/processed/final_elo_ratings_multi.json"
with open(out, "w") as f:
    json.dump({t: round(r, 2) for t, r in ratings.items()}, f, indent=2, ensure_ascii=False)
print(f"Guardado: {out}")

# Top 25
top = sorted(ratings.items(), key=lambda kv: -kv[1])[:25]
print(f"\n🏆 TOP 25 con Champions integrada:")
for i, (t, r) in enumerate(top, 1):
    print(f"  {i:2d}. {t:25s} {r:7.1f}")
