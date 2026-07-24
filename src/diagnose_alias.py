"""Detecta equipos FootballData sin match en Understat para ampliar TEAM_ALIAS."""
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent))
from fase2_full_pipeline import TEAM_ALIAS_FD_TO_UND, LEAGUE_MAP

ROOT = Path("/home/Projects/futbol_elo_poisson")
matches = pd.read_parquet(ROOT / "data/processed/matches_all.parquet")
und = pd.read_parquet(ROOT / "data/raw/understat_full/understat_schedule_5big.parquet")

# Aplico alias actual
matches["_home"] = matches["home_team"].apply(lambda t: TEAM_ALIAS_FD_TO_UND.get(t, t))
matches["_away"] = matches["away_team"].apply(lambda t: TEAM_ALIAS_FD_TO_UND.get(t, t))
matches["_league"] = matches["league"].map(LEAGUE_MAP)
matches["_date"] = pd.to_datetime(matches["date"]).dt.date

und["_date"] = pd.to_datetime(und["date"]).dt.date
und = und.rename(columns={"_league_norm": "_league"})

# Merge y ver qué NO matchea
m = matches.merge(
    und[["_league", "_date", "home_team", "away_team", "home_xg"]].rename(
        columns={"home_team": "_home", "away_team": "_away"}),
    on=["_league", "_date", "_home", "_away"], how="left"
)
no_match = m[m["home_xg"].isna()]
print(f"NO match: {len(no_match):,}/{len(m):,} ({len(no_match)/len(m)*100:.1f}%)")

# Por liga
print("\nNo-match por liga:")
print(no_match.groupby("_league").size().to_string())

# Nombres FD únicos sin match
fd_teams_no_match = set(no_match["home_team"].tolist() + no_match["away_team"].tolist())
und_teams_all = set(und["home_team"].tolist() + und["away_team"].tolist())
missing = sorted([t for t in fd_teams_no_match if TEAM_ALIAS_FD_TO_UND.get(t, t) not in und_teams_all])
print(f"\nEquipos FD sin match candidato en Understat: {len(missing)}")
for t in missing[:50]:
    # Sugerir Understat parecido
    candidates = [u for u in und_teams_all if t.lower()[:4] in u.lower() or u.lower()[:4] in t.lower()]
    print(f"  {t:30s}  →  candidatos: {candidates[:3]}")
