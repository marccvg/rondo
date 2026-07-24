"""
MINI paso 3 — Regresión Poisson (GLM).

Aprende matemáticamente:
  log(λ_home) = α_H + β_H * elo_diff
  log(λ_away) = α_A + β_A * elo_diff
Los coeficientes α y β los aprende STATSMODELS por máxima verosimilitud.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
from pathlib import Path

INITIAL, K, HOME_ADV = 1500.0, 20.0, 65.0

def expected(rh, ra):
    return 1.0 / (1.0 + 10 ** ((ra - (rh + HOME_ADV)) / 400.0))

def gd_mult(gd):
    g = abs(gd)
    if g <= 1: return 1.0
    if g == 2: return 1.5
    return (11.0 + g) / 8.0

# Reconstruir historia ELO (necesitamos elo PRE-partido, no final)
df = pd.read_parquet("/home/Projects/futbol_elo_poisson/mini/mini_matches.parquet")
df = df.sort_values("date").reset_index(drop=True)
ratings = {}

def get(t): return ratings.get(t, INITIAL)

elo_h, elo_a = [], []
for _, row in df.iterrows():
    rh_before = get(row["home_team"])
    ra_before = get(row["away_team"])
    elo_h.append(rh_before)
    elo_a.append(ra_before)
    actual = 1.0 if row["home_goals"] > row["away_goals"] else (0.0 if row["home_goals"] < row["away_goals"] else 0.5)
    exp = expected(rh_before, ra_before)
    m = gd_mult(row["home_goals"] - row["away_goals"])
    delta = K * m * (actual - exp)
    ratings[row["home_team"]] = rh_before + delta
    ratings[row["away_team"]] = ra_before - delta

df["elo_home"] = elo_h
df["elo_away"] = elo_a
df["elo_diff"] = df["elo_home"] - df["elo_away"]

print("=" * 60)
print("  MINI paso 3 — POISSON GLM")
print("=" * 60)

# Split train
train = df[df["date"] < "2023-07-01"].copy()
print(f"\nEntrenamiento con {len(train):,} partidos (temporadas 17/18 - 22/23).")

# Modelo home
X = sm.add_constant(train[["elo_diff"]])
mh = sm.GLM(train["home_goals"].astype(float), X, family=sm.families.Poisson()).fit()
ma = sm.GLM(train["away_goals"].astype(float), X, family=sm.families.Poisson()).fit()

print("\n─── Modelo goles HOME ───")
print(f"  Ecuación aprendida:  log(λ_home) = {mh.params['const']:.4f} + {mh.params['elo_diff']:.6f} · elo_diff")
print(f"  Interpretación con NÚMEROS:")
print(f"    Si elo_diff = 0 (equipos igualados):  λ_home = exp({mh.params['const']:.4f}) = {np.exp(mh.params['const']):.3f} goles esperados")
print(f"    Si elo_diff = +100 (home mejor):       λ_home = exp({mh.params['const']:.4f} + {mh.params['elo_diff']:.6f}*100) = {np.exp(mh.params['const'] + mh.params['elo_diff']*100):.3f}")
print(f"    Si elo_diff = +200:                    λ_home = {np.exp(mh.params['const'] + mh.params['elo_diff']*200):.3f}")
print(f"    Si elo_diff = -200 (visitante mejor):  λ_home = {np.exp(mh.params['const'] + mh.params['elo_diff']*(-200)):.3f}")

print("\n─── Modelo goles AWAY ───")
print(f"  Ecuación aprendida:  log(λ_away) = {ma.params['const']:.4f} + {ma.params['elo_diff']:.6f} · elo_diff")
print(f"  Si elo_diff = 0:    λ_away = exp({ma.params['const']:.4f}) = {np.exp(ma.params['const']):.3f}")
print(f"  Si elo_diff = +200: λ_away = {np.exp(ma.params['const'] + ma.params['elo_diff']*200):.3f}")
print(f"  Si elo_diff = -200: λ_away = {np.exp(ma.params['const'] + ma.params['elo_diff']*(-200)):.3f}")

print("\n─── Home advantage aprendido implícitamente ───")
print(f"  Cuando ELO igualado: 1.47 goles home vs {np.exp(ma.params['const']):.2f} away.")
print(f"  El home tiene ventaja de ~{(np.exp(mh.params['const']) / np.exp(ma.params['const']) - 1)*100:.0f}% en goles esperados.")

# Guardar df con ELO + params
df.to_parquet("/home/Projects/futbol_elo_poisson/mini/mini_matches_with_elo.parquet")
params = {
    "home_intercept": float(mh.params["const"]),
    "home_beta_elo": float(mh.params["elo_diff"]),
    "away_intercept": float(ma.params["const"]),
    "away_beta_elo": float(ma.params["elo_diff"]),
}
with open("/home/Projects/futbol_elo_poisson/mini/mini_poisson_params.json", "w") as f:
    json.dump(params, f, indent=2)
print(f"\nCoeficientes guardados.")
