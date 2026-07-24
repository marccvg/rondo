"""
Predicción unificada 5 competiciones 2026/27:
- Champions League
- Europa League
- Conference League
- LaLiga
- Premier League

Sobre ELO cross-league recomputado desde matches_all_multi.parquet (23.537 partidos
cronológicos 2017-2026).
"""
from __future__ import annotations
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/Projects/futbol_elo_poisson")

POISSON = {"home_intercept": 0.3883, "home_beta": 0.001842,
           "away_intercept": 0.1763, "away_beta": -0.001824}

ELO_DEFAULT = 1500.0
N_SIMS = 5000
rng = np.random.default_rng(42)


def load_elos() -> dict:
    return json.load(open(ROOT / "data/processed/final_elo_ratings_multi.json"))


def get_elo(team: str, elos: dict) -> float:
    return elos.get(team, ELO_DEFAULT)


def sim_match(eh, ea, neutral=False):
    diff = eh - ea
    lh = np.exp(POISSON["home_intercept"] + POISSON["home_beta"] * diff)
    la = np.exp(POISSON["away_intercept"] + POISSON["away_beta"] * diff)
    if neutral:
        avg = np.sqrt(lh * la)
        lh, la = avg, avg
    return rng.poisson(lh), rng.poisson(la)


# --- League round-robin doble ---
def sim_league(teams, elos, n_sims=N_SIMS):
    champion_c = Counter()
    top4_c = Counter()
    releg_c = Counter()
    for _ in range(n_sims):
        pts = defaultdict(int)
        for i in range(len(teams)):
            for j in range(len(teams)):
                if i == j: continue
                h, a = teams[i], teams[j]
                gh, ga = sim_match(get_elo(h, elos), get_elo(a, elos))
                if gh > ga: pts[h] += 3
                elif gh < ga: pts[a] += 3
                else: pts[h] += 1; pts[a] += 1
        ranked = sorted(teams, key=lambda t: -pts[t])
        champion_c[ranked[0]] += 1
        for t in ranked[:4]: top4_c[t] += 1
        for t in ranked[-3:]: releg_c[t] += 1
    return champion_c, top4_c, releg_c


# --- KO cup format (grupos 8 + bracket) ---
def sim_cup_32_teams(teams, elos, n_sims=N_SIMS):
    """Formato clásico: 32 equipos, 8 grupos de 4, KO. Sortear grupos aleatorio."""
    teams = list(teams)
    if len(teams) < 32:
        # Rellenar con equipos ELO más bajos para llegar a 32
        pool_low = [t for t in elos if t not in teams]
        teams += pool_low[:32 - len(teams)]
    teams = teams[:32]

    # 4 pots según ELO
    teams_sorted = sorted(teams, key=lambda t: -get_elo(t, elos))
    pots = [teams_sorted[i*8:(i+1)*8] for i in range(4)]

    champion_c = Counter()
    round16_c = Counter()
    quarters_c = Counter()
    semis_c = Counter()
    final_c = Counter()

    for _ in range(n_sims):
        # Sorteo grupos: cada grupo 1 de cada pot
        pots_copy = [list(pot) for pot in pots]
        for pot in pots_copy: rng.shuffle(pot)
        groups = {chr(ord("A")+gi): [pots_copy[p][gi] for p in range(4)] for gi in range(8)}

        # Fase grupos - todos vs todos ida/vuelta
        group_standings = {}
        for letter, gteams in groups.items():
            pts = defaultdict(int); gf = defaultdict(int); ga = defaultdict(int)
            for i in range(4):
                for j in range(4):
                    if i == j: continue
                    h, a = gteams[i], gteams[j]
                    gh, gag = sim_match(get_elo(h, elos), get_elo(a, elos))
                    gf[h] += gh; gf[a] += gag
                    ga[h] += gag; ga[a] += gh
                    if gh > gag: pts[h] += 3
                    elif gh < gag: pts[a] += 3
                    else: pts[h] += 1; pts[a] += 1
            ranked = sorted(gteams, key=lambda t: (-pts[t], -(gf[t]-ga[t]), -gf[t]))
            group_standings[letter] = ranked

        top1 = [group_standings[l][0] for l in "ABCDEFGH"]
        top2 = [group_standings[l][1] for l in "ABCDEFGH"]
        for t in top1 + top2: round16_c[t] += 1

        # Octavos: 1º vs 2º cross grupos
        seconds = list(top2); rng.shuffle(seconds)
        pairs = list(zip(top1, seconds))
        cuartos = []
        for h, a in pairs:
            g1h, g1a = sim_match(get_elo(h, elos), get_elo(a, elos))
            g2h, g2a = sim_match(get_elo(a, elos), get_elo(h, elos))
            agg_h = g1h + g2a; agg_a = g1a + g2h
            winner = h if agg_h > agg_a else (a if agg_a > agg_h else (h if rng.random() < 0.5 else a))
            cuartos.append(winner); quarters_c[winner] += 1

        rng.shuffle(cuartos)
        semi_teams = []
        for k in range(4):
            h, a = cuartos[2*k], cuartos[2*k+1]
            g1h, g1a = sim_match(get_elo(h, elos), get_elo(a, elos))
            g2h, g2a = sim_match(get_elo(a, elos), get_elo(h, elos))
            agg_h = g1h + g2a; agg_a = g1a + g2h
            winner = h if agg_h > agg_a else (a if agg_a > agg_h else (h if rng.random() < 0.5 else a))
            semi_teams.append(winner); semis_c[winner] += 1

        rng.shuffle(semi_teams)
        finalists = []
        for k in range(2):
            h, a = semi_teams[2*k], semi_teams[2*k+1]
            g1h, g1a = sim_match(get_elo(h, elos), get_elo(a, elos))
            g2h, g2a = sim_match(get_elo(a, elos), get_elo(h, elos))
            agg_h = g1h + g2a; agg_a = g1a + g2h
            winner = h if agg_h > agg_a else (a if agg_a > agg_h else (h if rng.random() < 0.5 else a))
            finalists.append(winner); final_c[winner] += 1

        # Final en campo neutral
        h, a = finalists
        gh, ga = sim_match(get_elo(h, elos), get_elo(a, elos), neutral=True)
        winner = h if gh > ga else (a if ga > gh else (h if rng.random() < 0.5 else a))
        champion_c[winner] += 1

    return champion_c, round16_c, quarters_c, semis_c, final_c


# --- Composiciones equipos 26/27 ---

def teams_from_last_season(df: pd.DataFrame, league: str) -> list:
    sub = df[(df["competition"] == league) & (df["season"] == "2025-26")]
    return sorted(set(sub["home_team"]) | set(sub["away_team"]))


def print_league_result(name, champion_c, top4_c, releg_c, n_sims=N_SIMS, top_n=10):
    print(f"\n{'━'*70}\n🏆  {name} 2026/27  ({n_sims:,} simulaciones)\n{'━'*70}")
    print("\nProbabilidad CAMPEÓN:")
    for t, c in champion_c.most_common(top_n):
        pct = c/n_sims*100
        bar = "█"*int(pct/2)
        print(f"  {t:22s}  {pct:5.1f}%  {bar}")
    print("\nProbabilidad TOP 4:")
    for t, c in top4_c.most_common(6):
        print(f"  {t:22s}  {c/n_sims*100:5.1f}%")
    print("\nProbabilidad DESCENSO (bottom 3):")
    for t, c in releg_c.most_common(6):
        print(f"  {t:22s}  {c/n_sims*100:5.1f}%")


def print_cup_result(name, champion_c, n_sims=N_SIMS, top_n=15):
    print(f"\n{'━'*70}\n🏆  {name} 2026/27  ({n_sims:,} simulaciones)\n{'━'*70}")
    print("\nProbabilidad CAMPEÓN:")
    for i, (t, c) in enumerate(champion_c.most_common(top_n), 1):
        pct = c/n_sims*100
        icon = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        bar = "█"*int(pct/2)
        print(f"  {icon} {i:2d}. {t:22s}  {pct:5.1f}%  {bar}")


# --- Composiciones cup 26/27 basadas en ELO ---
# Champions: top 32 ELO europeos que compiten
CHAMPIONS_POOL = [
    "Bayern Munich", "Arsenal", "Paris SG", "Barcelona", "Real Madrid",
    "Sp Lisbon", "Inter", "Benfica", "Man City", "Porto",
    "Dortmund", "Roma", "Man United", "Liverpool", "Leverkusen",
    "Napoli", "Juventus", "Ath Madrid", "Sp Braga", "Aston Villa",
    "Marseille", "Monaco", "AC Milan", "Milan", "Ath Bilbao",
    "PSV", "AFC Ajax", "Feyenoord", "Club Brugge KV", "Celtic FC",
    "FC Red Bull Salzburg", "Slavia Prague",
]

EUROPA_POOL = [
    "Villarreal", "Real Sociedad", "Betis", "Tottenham", "Chelsea",
    "Newcastle", "Fiorentina", "Lazio", "Como", "Lens", "Lille",
    "Nice", "Marseille", "Fenerbahçe", "Beşiktaş", "Rangers",
    "Celtic FC", "Ajax", "PSV", "Feyenoord", "AZ", "Twente",
    "Anderlecht", "Genk", "Sturm Graz", "Rapid Wien",
    "Braga", "Sp Braga", "AEK Athen", "Olympiacos", "Panathinaikos",
    "Union Saint-Gilloise",
]

CONFERENCE_POOL = [
    "Fiorentina", "Chelsea", "Legia Warszawa", "Basaksehir",
    "Rapid Wien", "Molde", "Salzburg", "Basel", "FC Basel 1893",
    "Slovan Bratislava", "APOEL Nikosia", "Genk", "Rijeka",
    "Malmö FF", "Sivasspor", "Copenhagen", "Djurgården",
    "Vitesse", "Trabzonspor", "Anderlecht", "Union Saint-Gilloise",
    "Riga", "Häcken", "Twente", "Bodø/Glimt", "FK Bodø/Glimt",
    "Astana", "APOEL", "Ludogorets", "Slavia Prague", "Sparta Praha",
    "Universitatea Craiova",
]


def main():
    elos = load_elos()
    df = pd.read_parquet(ROOT / "data/processed/matches_all_multi.parquet")

    print("═"*70)
    print("  PREDICCIONES 2026/27 — 5 COMPETICIONES")
    print("═"*70)
    print(f"\nBase: {len(df):,} partidos cronológicos 2017-2026, ELO cross-league.")
    print(f"Método: Monte Carlo N={N_SIMS:,} con Poisson GLM sobre ELO diff.")

    # LaLiga
    laliga_teams = teams_from_last_season(df, "domestic_laliga")
    print(f"\n📋 LaLiga 26/27 (equipos de 25/26): {len(laliga_teams)}")
    c, t4, r = sim_league(laliga_teams, elos)
    print_league_result("LaLiga", c, t4, r)

    # Premier
    premier_teams = teams_from_last_season(df, "domestic_premier")
    print(f"\n📋 Premier 26/27 (equipos de 25/26): {len(premier_teams)}")
    c, t4, r = sim_league(premier_teams, elos)
    print_league_result("Premier League", c, t4, r)

    # Champions
    print(f"\n📋 Champions League 26/27: 32 equipos top europeos por ELO")
    c, r16, qtr, sem, fin = sim_cup_32_teams(CHAMPIONS_POOL, elos)
    print_cup_result("Champions League", c)

    # Europa
    print(f"\n📋 Europa League 26/27: 32 equipos second-tier europeos por ELO")
    c, r16, qtr, sem, fin = sim_cup_32_teams(EUROPA_POOL, elos)
    print_cup_result("Europa League", c)

    # Conference
    print(f"\n📋 Conference League 26/27: 32 equipos third-tier europeos por ELO")
    c, r16, qtr, sem, fin = sim_cup_32_teams(CONFERENCE_POOL, elos)
    print_cup_result("Conference League", c)


if __name__ == "__main__":
    main()
