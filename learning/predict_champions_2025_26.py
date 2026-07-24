"""
Predicción Champions League 2025/26.

Usa el modelo Poisson GLM entrenado sobre el histórico ELO, con formato
CLÁSICO de 32 equipos + 8 grupos de 4 (por claridad pedagógica — el formato
real 2025/26 es el nuevo "Swiss" de 36 equipos, mucho más complejo de sortear).

Flujo:
  1. Top 32 equipos europeos por ELO actual
  2. Bombos (Pot 1-4) por ELO
  3. Sorteo aleatorio grupos respetando bombos
  4. Simulación Monte Carlo grupos + KO (10.000 sims)
  5. Probabilidad final de ganar la Champions
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path("/home/Projects/futbol_elo_poisson")
sys.path.insert(0, str(ROOT / "src"))

# Parámetros del modelo Poisson GLM aprendidos en Fase 1
# log(λ_home) = 0.3883 + 0.001842 · elo_diff
# log(λ_away) = 0.1763 - 0.001824 · elo_diff
POISSON = {
    "home_intercept": 0.3883,
    "home_beta": 0.001842,
    "away_intercept": 0.1763,
    "away_beta": -0.001824,
}


def dc_tau(i, j, lh, la, rho):
    if i == 0 and j == 0: return 1.0 - lh * la * rho
    if i == 0 and j == 1: return 1.0 + lh * rho
    if i == 1 and j == 0: return 1.0 + la * rho
    if i == 1 and j == 1: return 1.0 - rho
    return 1.0


def score_matrix(lh, la, rho=-0.03, size=9):
    ph = poisson.pmf(np.arange(size), lh)
    pa = poisson.pmf(np.arange(size), la)
    m = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            m[i, j] = ph[i] * pa[j] * dc_tau(i, j, lh, la, rho)
    return m / m.sum()


def simulate_match(elo_home, elo_away, neutral=False, rng=None):
    """Devuelve (goles_home, goles_away) sampleado según el modelo.
    neutral=True quita home advantage (aplica en KO neutral venues idealmente,
    aunque en formato clásico Champions se juega ida+vuelta con local diferente)."""
    if rng is None:
        rng = np.random.default_rng()
    diff = elo_home - elo_away
    lh = np.exp(POISSON["home_intercept"] + POISSON["home_beta"] * diff)
    la = np.exp(POISSON["away_intercept"] + POISSON["away_beta"] * diff)
    if neutral:
        # elimina home advantage: reduce lh y sube la a media geométrica
        avg = np.sqrt(lh * la)
        lh, la = avg, avg
    gh = rng.poisson(lh)
    ga = rng.poisson(la)
    return gh, ga


def load_elos():
    # Ratings finales calculados en mini/02_elo_paso_a_paso.py
    p = ROOT / "mini" / "mini_final_elo.json"
    if not p.exists():
        raise SystemExit(f"Falta {p}. Ejecuta antes: venv/bin/python mini/02_elo_paso_a_paso.py")
    return json.load(open(p))


# Top 32 equipos europeos que TÍPICAMENTE juegan Champions.
# Escogidos por ELO + presencia habitual en la competición.
CHAMPIONS_CANDIDATES = [
    "Paris SG", "Inter", "Bayern Munich", "Barcelona", "Real Madrid",
    "Liverpool", "Man City", "Arsenal", "Leverkusen", "Napoli",
    "Atalanta", "Juventus", "Ath Madrid", "Roma", "Dortmund",
    "Chelsea", "Ath Bilbao", "Villarreal", "Newcastle", "Tottenham",
    "AC Milan", "Marseille", "Monaco", "Sociedad", "Lazio",
    "Nice", "Betis", "Frankfurt", "Ein Frankfurt", "Aston Villa",
    "Fiorentina", "Nott'm Forest", "Wolves", "Brighton", "Bologna",
    "Girona", "Lens", "Lille", "RB Leipzig", "Sporting",
    "Benfica", "Porto", "Ajax", "PSV", "Feyenoord",
    "Club Brugge", "Slavia Prague", "Salzburg", "Celtic", "Rangers",
]


def format_percent(v):
    return f"{v*100:5.1f}%"


def main():
    print("═" * 78)
    print("  PREDICCIÓN CHAMPIONS LEAGUE 2025/26 (formato clásico 32 equipos)")
    print("═" * 78)

    elos = load_elos()
    # Filtrar solo los que tenemos ELO y quedarnos con los 32 top
    available = [(t, elos.get(t, 1500)) for t in CHAMPIONS_CANDIDATES]
    available = [(t, r) for t, r in available if t in elos]
    available.sort(key=lambda x: -x[1])
    teams32 = [t for t, _ in available[:32]]

    print(f"\n32 equipos participantes (ordenados por ELO actual):\n")
    for i, t in enumerate(teams32, 1):
        print(f"  {i:2d}. {t:25s} ELO {elos[t]:.0f}")

    # Bombos: 8 mejores en Pot1, siguientes 8 en Pot2, etc.
    pots = [teams32[i*8:(i+1)*8] for i in range(4)]

    print(f"\n╔═══ BOMBOS ═══╗")
    for pi, pot in enumerate(pots, 1):
        print(f"\n  🥇 Pot {pi}:")
        for t in pot:
            print(f"      {t:25s} (ELO {elos[t]:.0f})")

    # Sorteo aleatorio: 8 grupos, cada uno con 1 de cada pot
    random.seed(42)
    rng = np.random.default_rng(42)
    pots_copy = [pot.copy() for pot in pots]
    for pot in pots_copy:
        random.shuffle(pot)
    groups = {}
    for gi in range(8):
        letter = chr(ord("A") + gi)
        groups[letter] = [pots_copy[p][gi] for p in range(4)]

    print(f"\n\n╔═══ SORTEO DE GRUPOS (seed=42) ═══╗\n")
    for letter, teams in groups.items():
        print(f"  Grupo {letter}:")
        for t in teams:
            print(f"    {t:25s} ELO {elos[t]:.0f}")

    # Simulación Monte Carlo
    N_SIMS = 10000
    print(f"\n\n═══ SIMULACIÓN MONTE CARLO ({N_SIMS:,} torneos) ═══\n")

    group_first_c = Counter()
    group_second_c = Counter()
    quarterfinals_c = Counter()
    semifinals_c = Counter()
    finals_c = Counter()
    champions_c = Counter()

    for sim in range(N_SIMS):
        # === FASE DE GRUPOS ===
        group_standings = {}
        for letter, teams in groups.items():
            pts = defaultdict(int)
            gf = defaultdict(int)
            ga = defaultdict(int)
            # 6 partidos por grupo (todos contra todos ida y vuelta)
            for i in range(4):
                for j in range(4):
                    if i == j: continue
                    home, away = teams[i], teams[j]
                    gh, gag = simulate_match(elos[home], elos[away], rng=rng)
                    gf[home] += gh; gf[away] += gag
                    ga[home] += gag; ga[away] += gh
                    if gh > gag: pts[home] += 3
                    elif gh < gag: pts[away] += 3
                    else: pts[home] += 1; pts[away] += 1
            ranked = sorted(teams, key=lambda t: (-pts[t], -(gf[t]-ga[t]), -gf[t]))
            group_standings[letter] = ranked

        # Contar clasificados
        top1_this_sim = [group_standings[l][0] for l in "ABCDEFGH"]
        top2_this_sim = [group_standings[l][1] for l in "ABCDEFGH"]
        for t in top1_this_sim: group_first_c[t] += 1
        for t in top2_this_sim: group_second_c[t] += 1
        classified = top1_this_sim + top2_this_sim
        for t in classified: quarterfinals_c[t] += 0  # inicializar

        # === OCTAVOS (cross groups: 1º grupo vs 2º grupo diferente, sorteo random) ===
        seconds_shuffled = top2_this_sim.copy()
        rng.shuffle(seconds_shuffled)
        # Evitar mismo grupo (con este random simple puede ocurrir, forzamos re-shuffle)
        for _ in range(20):
            if all(top1_this_sim[i][:5] != seconds_shuffled[i][:5] or
                   top1_this_sim.index(top1_this_sim[i]) != top2_this_sim.index(seconds_shuffled[i])
                   for i in range(8)):
                break
            rng.shuffle(seconds_shuffled)

        octavos_pairs = list(zip(top1_this_sim, seconds_shuffled))
        cuartos_teams = []
        for h, a in octavos_pairs:
            # eliminatoria ida + vuelta
            g1h, g1a = simulate_match(elos[h], elos[a], rng=rng)
            g2h, g2a = simulate_match(elos[a], elos[h], rng=rng)
            agg_h = g1h + g2a
            agg_a = g1a + g2h
            if agg_h > agg_a: winner = h
            elif agg_a > agg_h: winner = a
            else:  # empate agregado, moneda al aire
                winner = h if rng.random() < 0.5 else a
            cuartos_teams.append(winner)
            quarterfinals_c[winner] += 1

        # === CUARTOS ===
        rng.shuffle(cuartos_teams)
        semis_teams = []
        for k in range(4):
            h, a = cuartos_teams[2*k], cuartos_teams[2*k+1]
            g1h, g1a = simulate_match(elos[h], elos[a], rng=rng)
            g2h, g2a = simulate_match(elos[a], elos[h], rng=rng)
            agg_h = g1h + g2a
            agg_a = g1a + g2h
            winner = h if agg_h > agg_a else (a if agg_a > agg_h else (h if rng.random() < 0.5 else a))
            semis_teams.append(winner)
            semifinals_c[winner] += 1

        # === SEMIS ===
        rng.shuffle(semis_teams)
        finalists = []
        for k in range(2):
            h, a = semis_teams[2*k], semis_teams[2*k+1]
            g1h, g1a = simulate_match(elos[h], elos[a], rng=rng)
            g2h, g2a = simulate_match(elos[a], elos[h], rng=rng)
            agg_h = g1h + g2a
            agg_a = g1a + g2h
            winner = h if agg_h > agg_a else (a if agg_a > agg_h else (h if rng.random() < 0.5 else a))
            finalists.append(winner)
            finals_c[winner] += 1

        # === FINAL === (partido único, campo neutral)
        h, a = finalists
        gh, ga = simulate_match(elos[h], elos[a], neutral=True, rng=rng)
        while gh == ga:
            # penales aleatorios
            winner = h if rng.random() < 0.5 else a
            break
        else:
            winner = h if gh > ga else a
        champions_c[winner] += 1

    # === RESULTADOS ===
    print(f"\n╔═══ PROBABILIDAD DE GANAR EL GRUPO ═══╗\n")
    for t, c in group_first_c.most_common(15):
        print(f"  {t:25s} {format_percent(c/N_SIMS)}")

    print(f"\n╔═══ PROBABILIDAD DE CLASIFICARSE A OCTAVOS (1º o 2º) ═══╗\n")
    combined = Counter()
    for t in teams32:
        combined[t] = group_first_c[t] + group_second_c[t]
    for t, c in combined.most_common(20):
        print(f"  {t:25s} {format_percent(c/N_SIMS)}")

    print(f"\n╔═══ PROBABILIDAD DE LLEGAR A CUARTOS ═══╗\n")
    for t, c in quarterfinals_c.most_common(15):
        print(f"  {t:25s} {format_percent(c/N_SIMS)}")

    print(f"\n╔═══ PROBABILIDAD DE LLEGAR A SEMIFINALES ═══╗\n")
    for t, c in semifinals_c.most_common(15):
        print(f"  {t:25s} {format_percent(c/N_SIMS)}")

    print(f"\n╔═══ PROBABILIDAD DE LLEGAR A LA FINAL ═══╗\n")
    for t, c in finals_c.most_common(10):
        print(f"  {t:25s} {format_percent(c/N_SIMS)}")

    print(f"\n\n╔═══════════════════════════════════════════════╗")
    print(f"║   🏆  CAMPEÓN CHAMPIONS 2025/26 (predicción)    ║")
    print(f"╚═══════════════════════════════════════════════╝\n")
    for i, (t, c) in enumerate(champions_c.most_common(15), 1):
        icon = "🏆" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else "  "))
        print(f"  {icon} {i:2d}. {t:25s} {format_percent(c/N_SIMS)}")

    print(f"\n💡 Aviso importante:")
    print(f"   - Usa el modelo Poisson GLM SIMPLE (1 feature ELO), no el LightGBM completo.")
    print(f"     Motivo: LightGBM necesita las 61 features históricas del partido concreto,")
    print(f"     imposible de construir para un partido futuro sin muchos supuestos.")
    print(f"   - Formato clásico 32 equipos, aunque el REAL 2025/26 es Swiss 36 equipos.")
    print(f"   - El sorteo aleatorio es UNO — con otro seed, otros grupos, otro campeón probable.")
    print(f"     Para robustez habría que iterar sobre múltiples seeds de sorteo.")


if __name__ == "__main__":
    main()
