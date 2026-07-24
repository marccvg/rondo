"""
Compara nuestro ELO (solo ligas domésticas) vs ClubELO externo (con Champions/Europa).
Muestra el impacto del "cross-league calibration" que nos falta.
"""
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path("/home/Projects/futbol_elo_poisson")
sys.path.insert(0, str(ROOT / "src"))

# Snapshot ClubELO (julio 2025, con Champions/Europa integradas)
clubelo = pd.read_csv(ROOT / "data/raw/clubelo_snapshot_2025-07-24.csv")

# Nuestro ELO histórico: recomputar ELO hasta ANTES de la temporada 25/26
# (para comparar en la misma "fecha" que ClubELO)
from elo import EloState
df = pd.read_parquet(ROOT / "data/processed/matches_all.parquet")
df_pre_2526 = df[df["date"] < "2025-07-01"].sort_values("date").reset_index(drop=True)
state = EloState()
for _, row in df_pre_2526.iterrows():
    state.update(row["home_team"], row["away_team"], int(row["home_goals"]), int(row["away_goals"]))

our_elo = pd.DataFrame([
    {"team": t, "our_elo": round(r, 1)} for t, r in state.ratings.items()
]).sort_values("our_elo", ascending=False)

# Alias mapping (nuestro nombre → nombre ClubELO)
ALIAS = {
    "Paris SG": "Paris SG", "Bayern Munich": "Bayern", "Real Madrid": "Real Madrid",
    "Barcelona": "Barcelona", "Ath Madrid": "Atletico", "Ath Bilbao": "Bilbao",
    "Man City": "Man City", "Man United": "Man United", "Nott'm Forest": "Forest",
    "Leverkusen": "Leverkusen", "Inter": "Inter", "AC Milan": "AC Milan",
    "Napoli": "Napoli", "Roma": "Roma", "Juventus": "Juventus",
    "Atalanta": "Atalanta", "Dortmund": "Dortmund", "Liverpool": "Liverpool",
    "Arsenal": "Arsenal", "Chelsea": "Chelsea", "Tottenham": "Tottenham",
    "Newcastle": "Newcastle", "Aston Villa": "Aston Villa", "Brighton": "Brighton",
    "Brentford": "Brentford", "Bournemouth": "Bournemouth", "Everton": "Everton",
    "Crystal Palace": "Crystal Palace", "West Ham": "West Ham", "Fulham": "Fulham",
    "Wolves": "Wolves",
}

# Join
rows = []
for _, r in our_elo.iterrows():
    cl_name = ALIAS.get(r["team"])
    if not cl_name:
        continue
    cl_row = clubelo[clubelo["Club"] == cl_name]
    if cl_row.empty:
        continue
    rows.append({
        "team": r["team"],
        "our_elo": r["our_elo"],
        "clubelo_elo": round(cl_row.iloc[0]["Elo"], 1),
        "clubelo_rank": int(cl_row.iloc[0]["Rank"]),
        "diff": round(cl_row.iloc[0]["Elo"] - r["our_elo"], 1),
    })

cmp = pd.DataFrame(rows).sort_values("our_elo", ascending=False)
cmp["our_rank"] = range(1, len(cmp) + 1)
cmp = cmp[["team", "our_rank", "our_elo", "clubelo_rank", "clubelo_elo", "diff"]]
print("Comparativa top 20 — nuestro ELO (solo ligas) vs ClubELO (con Champions):\n")
print(cmp.head(20).to_string(index=False))

print("\n\nEquipos que ClubELO valora MUCHO MÁS que nuestro modelo (diff > 100):")
positive = cmp[cmp["diff"] > 100].sort_values("diff", ascending=False)
print(positive.to_string(index=False))

print("\n\nEquipos que ClubELO valora MUCHO MENOS que nuestro modelo (diff < -50):")
negative = cmp[cmp["diff"] < -50].sort_values("diff")
print(negative.to_string(index=False))
