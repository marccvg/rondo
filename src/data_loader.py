"""
Carga y normaliza los CSVs de Football-Data.co.uk en un DataFrame único.

Columnas normalizadas:
  date, league, season, home_team, away_team, home_goals, away_goals,
  result (H/D/A), pinnacle_home, pinnacle_draw, pinnacle_away (cuotas cierre)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path("/home/Projects/futbol_elo_poisson/data/raw")

LEAGUE_MAP = {
    "E0": "premier",
    "SP1": "laliga",
    "I1": "seriea",
    "D1": "bundesliga",
    "F1": "ligue1",
    "P1": "portugal",
}

# Football-Data.co.uk columnas de cuotas Pinnacle de cierre
PINNACLE_COLS = {
    "pinnacle_home": ["PSCH", "PCH", "PH"],  # historicamente cambiaron nombres
    "pinnacle_draw": ["PSCD", "PCD", "PD"],
    "pinnacle_away": ["PSCA", "PCA", "PA"],
}


def _pick_first_available(row: pd.Series, candidates: list[str]) -> float:
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return float("nan")


def _extract_pinnacle(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae cuotas Pinnacle sea cual sea el naming histórico."""
    for target, cands in PINNACLE_COLS.items():
        found = None
        for c in cands:
            if c in df.columns:
                found = c
                break
        df[target] = df[found] if found else float("nan")
    return df


def load_all_footballdata() -> pd.DataFrame:
    """Carga todos los CSVs de Football-Data en un solo DataFrame normalizado."""
    frames = []
    for csv in sorted((RAW / "footballdata").glob("*.csv")):
        stem = csv.stem  # e.g. "E0_2324"
        lg_code, season_code = stem.split("_")
        league = LEAGUE_MAP.get(lg_code, lg_code)
        # season_code "2324" -> "2023-24"
        season = f"20{season_code[:2]}-{season_code[2:]}"
        try:
            df = pd.read_csv(csv, encoding="latin-1", on_bad_lines="skip", low_memory=False)
        except Exception as e:
            print(f"[skip] {csv.name}: {e}")
            continue

        df = _extract_pinnacle(df)

        # Columnas mínimas
        needed = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
        if not needed.issubset(df.columns):
            print(f"[skip] {csv.name}: falta columnas {needed - set(df.columns)}")
            continue

        out = pd.DataFrame({
            "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
            "league": league,
            "season": season,
            "home_team": df["HomeTeam"].astype(str),
            "away_team": df["AwayTeam"].astype(str),
            "home_goals": pd.to_numeric(df["FTHG"], errors="coerce").astype("Int64"),
            "away_goals": pd.to_numeric(df["FTAG"], errors="coerce").astype("Int64"),
            "result": df["FTR"].astype(str),  # H / D / A
            "pinnacle_home": pd.to_numeric(df["pinnacle_home"], errors="coerce"),
            "pinnacle_draw": pd.to_numeric(df["pinnacle_draw"], errors="coerce"),
            "pinnacle_away": pd.to_numeric(df["pinnacle_away"], errors="coerce"),
        })
        # drop rows con NaT en date o goles missing
        out = out.dropna(subset=["date", "home_goals", "away_goals"])
        frames.append(out)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values("date").reset_index(drop=True)
    return all_df


def implied_probs_from_odds(odds_h: float, odds_d: float, odds_a: float) -> tuple[float, float, float]:
    """De cuotas decimales a probabilidades normalizadas (elimina margen casa)."""
    inv = [1 / odds_h, 1 / odds_d, 1 / odds_a]
    s = sum(inv)
    return inv[0] / s, inv[1] / s, inv[2] / s


if __name__ == "__main__":
    df = load_all_footballdata()
    print(f"Total partidos cargados: {len(df):,}")
    print(f"Rango fechas: {df['date'].min()} → {df['date'].max()}")
    print(f"Por liga:\n{df.groupby('league').size()}")
    print(f"Con cuotas Pinnacle: {df['pinnacle_home'].notna().sum():,} ({df['pinnacle_home'].notna().mean():.1%})")
    out = RAW.parent / "processed" / "matches_all.parquet"
    out.parent.mkdir(exist_ok=True)
    df.to_parquet(out)
    print(f"Guardado: {out}")
