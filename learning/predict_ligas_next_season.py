"""
Predicción LaLiga 2025/26 + Premier League 2025/26.

Mismo modelo Poisson GLM (elo_diff) usado en Champions predictor.
Formato: round-robin completo (38 jornadas), Monte Carlo 5000 simulaciones.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import poisson

ROOT = Path("/home/Projects/futbol_elo_poisson")

POISSON = {
    "home_intercept": 0.3883,
    "home_beta": 0.001842,
    "away_intercept": 0.1763,
    "away_beta": -0.001824,
}

# ELO por defecto para recién ascendidos (típicamente débiles primer año)
ELO_ASCENDIDO = 1420.0

# Composición LaLiga 25/26
LALIGA_25_26 = [
    # 17 que se salvaron 24/25
    "Real Madrid", "Barcelona", "Ath Madrid", "Villarreal", "Ath Bilbao",
    "Betis", "Celta", "Vallecano", "Osasuna", "Sociedad",
    "Mallorca", "Alaves", "Espanol", "Valencia", "Getafe",
    "Girona", "Sevilla",
    # 3 ascendidos
    "Elche", "Levante", "Real Oviedo",
]

# Composición Premier League 25/26
PREMIER_25_26 = [
    # 17 que se salvaron 24/25
    "Liverpool", "Arsenal", "Man City", "Chelsea", "Newcastle",
    "Aston Villa", "Nott'm Forest", "Brighton", "Bournemouth", "Brentford",
    "Fulham", "Crystal Palace", "Everton", "West Ham", "Man United",
    "Wolves", "Tottenham",
    # 3 ascendidos
    "Leeds", "Burnley", "Sunderland",
]


def simulate_match(elo_home, elo_away, rng):
    diff = elo_home - elo_away
    lh = np.exp(POISSON["home_intercept"] + POISSON["home_beta"] * diff)
    la = np.exp(POISSON["away_intercept"] + POISSON["away_beta"] * diff)
    return rng.poisson(lh), rng.poisson(la)


def get_elo(team, elos):
    e = elos.get(team)
    if e is None:
        return ELO_ASCENDIDO
    return e


def simulate_league(teams, elos, n_sims=5000):
    """Round-robin doble (todos vs todos ida/vuelta)."""
    rng = np.random.default_rng(42)

    champion_c = Counter()
    top4_c = Counter()
    releg_c = Counter()  # bottom 3 (LaLiga/Premier descienden 3)
    top10_c = Counter()

    for _ in range(n_sims):
        pts = defaultdict(int)
        for i in range(len(teams)):
            for j in range(len(teams)):
                if i == j:
                    continue
                home, away = teams[i], teams[j]
                eh, ea = get_elo(home, elos), get_elo(away, elos)
                gh, ga = simulate_match(eh, ea, rng)
                if gh > ga: pts[home] += 3
                elif gh < ga: pts[away] += 3
                else: pts[home] += 1; pts[away] += 1
        ranked = sorted(teams, key=lambda t: -pts[t])
        champion_c[ranked[0]] += 1
        for t in ranked[:4]: top4_c[t] += 1
        for t in ranked[:10]: top10_c[t] += 1
        for t in ranked[-3:]: releg_c[t] += 1

    return {
        "champion": champion_c,
        "top4": top4_c,
        "top10": top10_c,
        "relegation": releg_c,
        "n_sims": n_sims,
    }


def print_results(name, teams, elos, results):
    print(f"\n{'═' * 78}")
    print(f"   🏆  PREDICCIÓN {name} — 5.000 SIMULACIONES")
    print(f"{'═' * 78}")

    n = results["n_sims"]

    print(f"\n📋 20 equipos (ordenados por ELO):\n")
    with_elo = sorted([(t, get_elo(t, elos), t not in elos) for t in teams],
                      key=lambda x: -x[1])
    for i, (t, e, is_asc) in enumerate(with_elo, 1):
        note = " (ascendido, ELO por defecto)" if is_asc else ""
        print(f"  {i:2d}. {t:20s} {e:.0f}{note}")

    print(f"\n🏆 PROBABILIDAD CAMPEÓN:\n")
    for t, c in results["champion"].most_common(10):
        pct = c / n * 100
        bar = "█" * int(pct / 2)
        print(f"  {t:20s}  {pct:5.1f}%  {bar}")

    print(f"\n🥇 PROBABILIDAD TOP-4 (Champions):\n")
    for t, c in results["top4"].most_common(8):
        print(f"  {t:20s}  {c/n*100:5.1f}%")

    print(f"\n🎯 PROBABILIDAD TOP-10:\n")
    for t, c in results["top10"].most_common(12):
        print(f"  {t:20s}  {c/n*100:5.1f}%")

    print(f"\n📉 PROBABILIDAD DESCENSO (bottom 3):\n")
    for t, c in results["relegation"].most_common(8):
        print(f"  {t:20s}  {c/n*100:5.1f}%")


def main():
    elos_path = ROOT / "mini" / "mini_final_elo.json"
    if not elos_path.exists():
        print(f"Falta {elos_path}. Ejecuta: venv/bin/python mini/02_elo_paso_a_paso.py")
        return
    elos = json.load(open(elos_path))

    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  PREDICCIÓN DE LIGAS 2025/26 — LaLiga y Premier                          ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    print("║  Método: Poisson GLM (1 feature elo_diff) + Monte Carlo round-robin      ║")
    print("║  Ascendidos → ELO por defecto 1420 (débiles típicos primer año)          ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")

    # LALIGA
    r_lig = simulate_league(LALIGA_25_26, elos, n_sims=5000)
    print_results("LaLiga 2025/26", LALIGA_25_26, elos, r_lig)

    # PREMIER
    r_pre = simulate_league(PREMIER_25_26, elos, n_sims=5000)
    print_results("Premier League 2025/26", PREMIER_25_26, elos, r_pre)

    # Comparativa
    print(f"\n{'═' * 78}")
    print("  📊 COMPARATIVA CAMPEONES — LaLiga vs Premier")
    print(f"{'═' * 78}")
    print(f"\n{'LaLiga 25/26':<35}  {'Premier 25/26':<35}")
    print("─" * 78)
    liga_top5 = r_lig["champion"].most_common(5)
    pre_top5 = r_pre["champion"].most_common(5)
    for i in range(5):
        l = f"{liga_top5[i][0]} ({liga_top5[i][1]/5000*100:.1f}%)"
        p = f"{pre_top5[i][0]} ({pre_top5[i][1]/5000*100:.1f}%)"
        print(f"  {i+1}. {l:<32}  {i+1}. {p}")

    print(f"\n\n💡 AVISOS HONESTOS:")
    print(f"  • Modelo Poisson simple (1 feature ELO), no LightGBM (imposible sin features hist. de partido futuro).")
    print(f"  • Recién ascendidos (Elche, Levante, Oviedo, Leeds, Burnley, Sunderland) → ELO=1420 por defecto.")
    print(f"    Probablemente están infra-valorados vs realidad — hay campeones de segunda/championship fuertes.")
    print(f"  • Sistema round-robin doble asume calendario típico. No modelamos calendario real, Cup fixtures, etc.")
    print(f"  • Sin factores externos (lesiones, mercado enero, cambios entrenador). Modelo 'a fecha de hoy'.")


if __name__ == "__main__":
    main()
