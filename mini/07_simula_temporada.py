"""
MINI paso 7 — SIMULACIÓN MONTE CARLO de temporada LaLiga 2024-25.

Estrategia:
1. Tomar partidos REALES de LaLiga 24-25.
2. Para cada partido, calcular λ_home y λ_away con LGB.
3. Simular 10.000 veces cada partido → tabla acumulada.
4. Contar probabilidad campeón por equipo.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter, defaultdict

N_SIMS = 5000
FEATURES = ["elo_diff", "forma_home_gs5", "forma_home_gc5", "forma_away_gs5", "forma_away_gc5"]

df = pd.read_parquet("/home/Projects/futbol_elo_poisson/mini/mini_matches_with_features.parquet")
m_h = lgb.Booster(model_file="/home/Projects/futbol_elo_poisson/mini/mini_lgb_home.txt")
m_a = lgb.Booster(model_file="/home/Projects/futbol_elo_poisson/mini/mini_lgb_away.txt")

# LaLiga 2024-25 — usamos temporada test
laliga = df[(df["date"] >= "2024-07-01") & (df["date"] < "2025-07-01")].copy()
# Filtro heurístico por equipos históricamente LaLiga (los que aparecen suficientes veces)
teams_all = pd.concat([laliga["home_team"], laliga["away_team"]])
top_teams = teams_all.value_counts().head(20).index.tolist()
laliga = laliga[laliga["home_team"].isin(top_teams) & laliga["away_team"].isin(top_teams)]
print(f"Filtrado LaLiga 24/25: {len(laliga)} partidos, {len(top_teams)} equipos")

# Predicciones λ para cada partido
X = laliga[FEATURES]
laliga["lh"] = m_h.predict(X)
laliga["la"] = m_a.predict(X)

# Simulación Monte Carlo
rng = np.random.default_rng(42)
champion_counter = Counter()
top4_counter = Counter()
relegation_counter = Counter()

for sim in range(N_SIMS):
    puntos = defaultdict(int)
    for _, row in laliga.iterrows():
        gh = rng.poisson(row["lh"])
        ga = rng.poisson(row["la"])
        if gh > ga:
            puntos[row["home_team"]] += 3
        elif gh < ga:
            puntos[row["away_team"]] += 3
        else:
            puntos[row["home_team"]] += 1
            puntos[row["away_team"]] += 1
    ranked = sorted(puntos.items(), key=lambda kv: -kv[1])
    champion_counter[ranked[0][0]] += 1
    for team, _ in ranked[:4]:
        top4_counter[team] += 1
    for team, _ in ranked[-3:]:
        relegation_counter[team] += 1

print("\n" + "=" * 60)
print(f"  RESULTADOS TRAS {N_SIMS:,} SIMULACIONES")
print("=" * 60)

print("\n🏆 PROBABILIDAD DE SER CAMPEÓN")
for team, wins in champion_counter.most_common(10):
    print(f"  {team:20s}  {wins/N_SIMS*100:5.1f}%")

print("\n🥇 PROBABILIDAD DE TOP 4 (Champions League)")
for team, cnt in top4_counter.most_common(10):
    print(f"  {team:20s}  {cnt/N_SIMS*100:5.1f}%")

print("\n📉 PROBABILIDAD DE DESCENSO (3 últimos)")
for team, cnt in relegation_counter.most_common(8):
    print(f"  {team:20s}  {cnt/N_SIMS*100:5.1f}%")

# Comparar con resultado real
print("\n\n─── Ranking REAL final LaLiga 24-25 (según los datos) ───")
puntos_real = defaultdict(int)
for _, row in laliga.iterrows():
    if row["home_goals"] > row["away_goals"]:
        puntos_real[row["home_team"]] += 3
    elif row["home_goals"] < row["away_goals"]:
        puntos_real[row["away_team"]] += 3
    else:
        puntos_real[row["home_team"]] += 1
        puntos_real[row["away_team"]] += 1
ranked_real = sorted(puntos_real.items(), key=lambda kv: -kv[1])
for i, (t, p) in enumerate(ranked_real[:10], 1):
    prob_camp = champion_counter[t] / N_SIMS * 100
    print(f"  {i:2d}. {t:20s} {p:3d} pts  (modelo predecía campeón con {prob_camp:.1f}%)")
