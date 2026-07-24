"""
MINI paso 6 — Predecir un partido concreto de arriba abajo mostrando TODOS los cálculos.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import poisson

HOME_TEAM = "Real Madrid"
AWAY_TEAM = "Barcelona"

# Cargar todo
df = pd.read_parquet("/home/Projects/futbol_elo_poisson/mini/mini_matches_with_features.parquet")
params_poisson = json.load(open("/home/Projects/futbol_elo_poisson/mini/mini_poisson_params.json"))
elo_final = json.load(open("/home/Projects/futbol_elo_poisson/mini/mini_final_elo.json"))
m_h = lgb.Booster(model_file="/home/Projects/futbol_elo_poisson/mini/mini_lgb_home.txt")
m_a = lgb.Booster(model_file="/home/Projects/futbol_elo_poisson/mini/mini_lgb_away.txt")

print("=" * 70)
print(f"  MINI paso 6 — PREDICCIÓN COMPLETA: {HOME_TEAM} vs {AWAY_TEAM}")
print("=" * 70)

# 1. Ratings actuales
elo_h = elo_final.get(HOME_TEAM, 1500)
elo_a = elo_final.get(AWAY_TEAM, 1500)
elo_diff = elo_h - elo_a
print(f"\n1️⃣  RATINGS ELO (últimos disponibles)")
print(f"   {HOME_TEAM}:  {elo_h:.1f}")
print(f"   {AWAY_TEAM}:  {elo_a:.1f}")
print(f"   elo_diff (home - away):  {elo_diff:+.1f}")

# 2. Forma reciente (últimos 5 partidos cada uno)
def ultimos5_gs(team):
    m = df[(df["home_team"]==team)|(df["away_team"]==team)].sort_values("date").tail(5)
    gs = [(r["home_goals"] if r["home_team"]==team else r["away_goals"]) for _,r in m.iterrows()]
    gc = [(r["away_goals"] if r["home_team"]==team else r["home_goals"]) for _,r in m.iterrows()]
    return np.mean(gs), np.mean(gc), gs, gc

hg, hc, hg_list, hc_list = ultimos5_gs(HOME_TEAM)
ag, ac, ag_list, ac_list = ultimos5_gs(AWAY_TEAM)
print(f"\n2️⃣  FORMA ÚLTIMOS 5 (goles marcados/encajados)")
print(f"   {HOME_TEAM}:  marcados={hg_list} → media={hg:.2f}  |  encajados={hc_list} → media={hc:.2f}")
print(f"   {AWAY_TEAM}:  marcados={ag_list} → media={ag:.2f}  |  encajados={ac_list} → media={ac:.2f}")

# 3. Predicción con Poisson GLM (modelo lineal)
print(f"\n3️⃣  PREDICCIÓN con POISSON GLM (baseline lineal)")
log_lh = params_poisson["home_intercept"] + params_poisson["home_beta_elo"] * elo_diff
log_la = params_poisson["away_intercept"] + params_poisson["away_beta_elo"] * elo_diff
lh_poi = np.exp(log_lh)
la_poi = np.exp(log_la)
print(f"   log(λ_home) = {params_poisson['home_intercept']:.4f} + {params_poisson['home_beta_elo']:.6f} × {elo_diff:+.1f} = {log_lh:.4f}")
print(f"   log(λ_away) = {params_poisson['away_intercept']:.4f} + {params_poisson['away_beta_elo']:.6f} × {elo_diff:+.1f} = {log_la:.4f}")
print(f"   λ_home = exp({log_lh:.4f}) = {lh_poi:.3f} goles esperados home")
print(f"   λ_away = exp({log_la:.4f}) = {la_poi:.3f} goles esperados away")

# 4. Predicción con LightGBM
features = ["elo_diff", "forma_home_gs5", "forma_home_gc5", "forma_away_gs5", "forma_away_gc5"]
X = pd.DataFrame([{
    "elo_diff": elo_diff,
    "forma_home_gs5": hg,
    "forma_home_gc5": hc,
    "forma_away_gs5": ag,
    "forma_away_gc5": ac,
}])
lh_lgb = m_h.predict(X)[0]
la_lgb = m_a.predict(X)[0]
print(f"\n4️⃣  PREDICCIÓN con LIGHTGBM")
print(f"   λ_home = {lh_lgb:.3f}  (128 árboles votando)")
print(f"   λ_away = {la_lgb:.3f}")

# 5. Matriz Dixon-Coles usando LGB
def dc_tau(i, j, lh, la, rho):
    if i == 0 and j == 0: return 1.0 - lh * la * rho
    if i == 0 and j == 1: return 1.0 + lh * rho
    if i == 1 and j == 0: return 1.0 + la * rho
    if i == 1 and j == 1: return 1.0 - rho
    return 1.0

def matriz(lh, la, rho, size=9):
    p_h = poisson.pmf(np.arange(size), lh)
    p_a = poisson.pmf(np.arange(size), la)
    m = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            m[i, j] = p_h[i] * p_a[j] * dc_tau(i, j, lh, la, rho)
    return m / m.sum()

mat = matriz(lh_lgb, la_lgb, rho=-0.03)
print(f"\n5️⃣  MATRIZ RESULTADO EXACTO (LGB + Dixon-Coles rho=-0.03)")
print(f"        " + "  ".join(f"A={j}" for j in range(6)))
for i in range(6):
    row = "  ".join(f"{mat[i,j]:.3f}" for j in range(6))
    print(f"   H={i} {row}")

# 6. Mercados derivados
def mercados(m):
    n = m.shape[0]
    return {
        "H": sum(m[i,j] for i in range(n) for j in range(n) if i>j),
        "D": sum(m[i,i] for i in range(n)),
        "A": sum(m[i,j] for i in range(n) for j in range(n) if i<j),
        "over25": sum(m[i,j] for i in range(n) for j in range(n) if i+j > 2),
        "under25": sum(m[i,j] for i in range(n) for j in range(n) if i+j <= 2),
        "btts": sum(m[i,j] for i in range(1,n) for j in range(1,n)),
    }
mk = mercados(mat)
print(f"\n6️⃣  PROBABILIDADES por MERCADO")
print(f"   1X2:")
print(f"     P({HOME_TEAM} gana) = {mk['H']:.3f}  ({mk['H']*100:.1f}%)")
print(f"     P(Empate)          = {mk['D']:.3f}  ({mk['D']*100:.1f}%)")
print(f"     P({AWAY_TEAM} gana) = {mk['A']:.3f}  ({mk['A']*100:.1f}%)")
print(f"   Total goles:")
print(f"     P(Over 2.5)  = {mk['over25']:.3f}")
print(f"     P(Under 2.5) = {mk['under25']:.3f}")
print(f"   P(Ambos marcan) = {mk['btts']:.3f}")

# 7. Top 5 resultados más probables
print(f"\n7️⃣  TOP 5 RESULTADOS EXACTOS más probables")
flat = [(i, j, mat[i,j]) for i in range(9) for j in range(9)]
flat.sort(key=lambda x: -x[2])
for i, j, p in flat[:5]:
    print(f"     {HOME_TEAM} {i}-{j} {AWAY_TEAM}   P = {p:.4f}  ({p*100:.2f}%)")
