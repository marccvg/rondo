"""
ELO clásico para fútbol con ajuste de goal difference y ventaja local.

Referencia: FiveThirtyEight + ClubELO. K adaptativo por margen goles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


DEFAULT_K = 20.0
DEFAULT_HOME_ADV = 65.0  # equivalente Elo puntos, ~ ClubELO
DEFAULT_INITIAL = 1500.0


def expected_score(rating_a: float, rating_b: float, home_adv: float = 0.0) -> float:
    """Probabilidad esperada de que A gane vs B (con ventaja local sumada a A si aplica)."""
    diff = (rating_b - (rating_a + home_adv)) / 400.0
    return 1.0 / (1.0 + 10 ** diff)


def goal_diff_multiplier(goal_diff: int) -> float:
    """Multiplicador K por margen — recompensa mayor a victorias amplias."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0  # gd=3 → 1.75, gd=4 → 1.875, gd=5 → 2.0


def update_elo(
    rating_home: float,
    rating_away: float,
    home_goals: int,
    away_goals: int,
    k: float = DEFAULT_K,
    home_adv: float = DEFAULT_HOME_ADV,
) -> tuple[float, float]:
    """Actualiza ratings tras un partido. Devuelve (nuevo_home, nuevo_away)."""
    if home_goals > away_goals:
        actual_home = 1.0
    elif home_goals < away_goals:
        actual_home = 0.0
    else:
        actual_home = 0.5
    exp_home = expected_score(rating_home, rating_away, home_adv=home_adv)
    gd_mult = goal_diff_multiplier(home_goals - away_goals)
    delta = k * gd_mult * (actual_home - exp_home)
    return rating_home + delta, rating_away - delta


@dataclass
class EloState:
    """Mantiene ratings vivos por equipo."""
    ratings: dict[str, float] = field(default_factory=dict)
    initial: float = DEFAULT_INITIAL
    k: float = DEFAULT_K
    home_adv: float = DEFAULT_HOME_ADV

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.initial)

    def update(self, home_team: str, away_team: str, home_goals: int, away_goals: int) -> tuple[float, float]:
        rh_before = self.get(home_team)
        ra_before = self.get(away_team)
        rh_after, ra_after = update_elo(rh_before, ra_before, home_goals, away_goals, self.k, self.home_adv)
        self.ratings[home_team] = rh_after
        self.ratings[away_team] = ra_after
        return rh_before, ra_before  # ratings PRE-partido para features


def compute_elo_history(matches: pd.DataFrame, k: float = DEFAULT_K, home_adv: float = DEFAULT_HOME_ADV) -> pd.DataFrame:
    """
    Corre ELO recursivo sobre matches (ordenados por fecha).
    Añade columnas 'elo_home', 'elo_away', 'elo_prob_home' con ratings PRE-partido.
    """
    matches = matches.sort_values("date").reset_index(drop=True).copy()
    state = EloState(k=k, home_adv=home_adv)
    elo_home = []
    elo_away = []
    elo_prob = []
    for row in matches.itertuples(index=False):
        rh, ra = state.update(row.home_team, row.away_team, int(row.home_goals), int(row.away_goals))
        elo_home.append(rh)
        elo_away.append(ra)
        elo_prob.append(expected_score(rh, ra, home_adv=home_adv))
    matches["elo_home"] = elo_home
    matches["elo_away"] = elo_away
    matches["elo_prob_home"] = elo_prob
    return matches


if __name__ == "__main__":
    from pathlib import Path
    df = pd.read_parquet(Path("/home/Projects/futbol_elo_poisson/data/processed/matches_all.parquet"))
    out = compute_elo_history(df)
    print(f"Elo corrido sobre {len(out):,} partidos")
    print("Top 10 ratings finales:")
    state = EloState()
    for _, row in out.iterrows():
        state.update(row["home_team"], row["away_team"], int(row["home_goals"]), int(row["away_goals"]))
    top = sorted(state.ratings.items(), key=lambda kv: -kv[1])[:15]
    for team, r in top:
        print(f"  {team:25s} {r:7.1f}")
    save = Path("/home/Projects/futbol_elo_poisson/data/processed/matches_with_elo.parquet")
    out.to_parquet(save)
    print(f"\nGuardado: {save}")
