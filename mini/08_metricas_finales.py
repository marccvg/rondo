"""
MINI paso 8 — MÉTRICAS FINALES: comparar mini vs baselines.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import poisson

FEATURES = ["elo_diff", "forma_home_gs5", "forma_home_gc5", "forma_away_gs5", "forma_away_gc5"]
RESULT_TO_IDX = {"H": 0, "D": 1, "A": 2}


def dc_tau(i, j, lh, la, rho):
    if i == 0 and j == 0: return 1.0 - lh * la * rho
    if i == 0 and j == 1: return 1.0 + lh * rho
    if i == 1 and j == 0: return 1.0 + la * rho
    if i == 1 and j == 1: return 1.0 - rho
    return 1.0


def matriz(lh, la, rho=-0.03, size=9):
    p_h = poisson.pmf(np.arange(size), lh)
    p_a = poisson.pmf(np.arange(size), la)
    m = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            m[i, j] = p_h[i] * p_a[j] * dc_tau(i, j, lh, la, rho)
    return m / m.sum()


def m_to_1x2(m):
    n = m.shape[0]
    return (
        sum(m[i, j] for i in range(n) for j in range(n) if i > j),
        sum(m[i, i] for i in range(n)),
        sum(m[i, j] for i in range(n) for j in range(n) if i < j),
    )


def log_loss(y_true, probs, eps=1e-15):
    y_idx = np.array([RESULT_TO_IDX[r] for r in y_true])
    p = np.clip(probs[np.arange(len(y_idx)), y_idx], eps, 1.0)
    return -np.log(p).mean()


def rps(y_true, probs):
    y_idx = np.array([RESULT_TO_IDX[r] for r in y_true])
    n = len(y_idx)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y_idx] = 1.0
    cum_p = np.cumsum(probs[:, :2], axis=1)
    cum_y = np.cumsum(onehot[:, :2], axis=1)
    return ((cum_p - cum_y) ** 2).sum(axis=1).mean()


# Cargar test
df = pd.read_parquet("/home/Projects/futbol_elo_poisson/mini/mini_matches_with_features.parquet")
test = df[df["date"] >= "2024-07-01"].copy()
print(f"Test: {len(test):,} partidos (temporada 24-25)")
y_true = test["result"].to_numpy()

# 1. Uniforme
probs_unif = np.tile([1/3, 1/3, 1/3], (len(test), 1))

# 2. Home prior
train = df[df["date"] < "2023-07-01"]
counts = train["result"].value_counts(normalize=True)
probs_home = np.tile([counts["H"], counts["D"], counts["A"]], (len(test), 1))

# 3. ELO+Poisson+DC (mini pero mismo algoritmo que Fase 1 baseline)
import json
p = json.load(open("/home/Projects/futbol_elo_poisson/mini/mini_poisson_params.json"))
lh_p = np.exp(p["home_intercept"] + p["home_beta_elo"] * test["elo_diff"].to_numpy())
la_p = np.exp(p["away_intercept"] + p["away_beta_elo"] * test["elo_diff"].to_numpy())
probs_poi = np.zeros((len(test), 3))
for i, (lh, la) in enumerate(zip(lh_p, la_p)):
    m = matriz(lh, la, rho=-0.03)
    probs_poi[i] = m_to_1x2(m)

# 4. LightGBM+DC (mini)
m_h = lgb.Booster(model_file="/home/Projects/futbol_elo_poisson/mini/mini_lgb_home.txt")
m_a = lgb.Booster(model_file="/home/Projects/futbol_elo_poisson/mini/mini_lgb_away.txt")
lh_l = m_h.predict(test[FEATURES])
la_l = m_a.predict(test[FEATURES])
probs_lgb = np.zeros((len(test), 3))
for i, (lh, la) in enumerate(zip(lh_l, la_l)):
    m = matriz(lh, la, rho=-0.03)
    probs_lgb[i] = m_to_1x2(m)

# 5. Pinnacle (usando cuotas del matches_all — mini no las tiene, joineo)
big = pd.read_parquet("/home/Projects/futbol_elo_poisson/data/processed/matches_all.parquet")
big["_k"] = big["date"].astype(str) + "|" + big["home_team"] + "|" + big["away_team"]
test["_k"] = test["date"].astype(str) + "|" + test["home_team"] + "|" + test["away_team"]
big_test = big.set_index("_k")
test_j = test.merge(
    big[["_k", "pinnacle_home", "pinnacle_draw", "pinnacle_away"]],
    on="_k", how="left"
)
probs_pin = np.zeros((len(test_j), 3))
for i, r in enumerate(test_j.itertuples()):
    if pd.isna(r.pinnacle_home):
        probs_pin[i] = [1/3, 1/3, 1/3]
    else:
        inv = [1/r.pinnacle_home, 1/r.pinnacle_draw, 1/r.pinnacle_away]
        s = sum(inv)
        probs_pin[i] = [x/s for x in inv]

# Comparación
print("\n" + "=" * 70)
print("  MÉTRICAS TEST (temporada 24-25) — mini vs baseline grande vs Pinnacle")
print("=" * 70)
print(f"\n{'Modelo':<28}  {'log-loss':>9}  {'RPS':>7}  {'Comentario'}")
print("─" * 70)

for name, probs in [
    ("Uniforme", probs_unif),
    ("Home prior", probs_home),
    ("MINI: ELO+Poisson+DC", probs_poi),
    ("MINI: LightGBM+DC", probs_lgb),
    ("Pinnacle (techo)", probs_pin),
]:
    ll = log_loss(y_true, probs)
    r = rps(y_true, probs)
    print(f"{name:<28}  {ll:>9.4f}  {r:>7.4f}")

print("\n" + "=" * 70)
print("  COMPARACIÓN vs BASELINE GRANDE (Fase 1 del pipeline completo)")
print("=" * 70)
print(f"Baseline grande Fase 1: log-loss=0.9859, RPS=0.4015 (mismo test set)")
print(f"MINI ELO+Poisson+DC:    log-loss={log_loss(y_true, probs_poi):.4f}, RPS={rps(y_true, probs_poi):.4f}")
print(f"MINI LightGBM+DC:       log-loss={log_loss(y_true, probs_lgb):.4f}, RPS={rps(y_true, probs_lgb):.4f}")
print("\n→ Los mini deberían quedar CERCA del baseline grande porque usan")
print("  esencialmente los mismos inputs (ELO + goles rolling). El pipeline")
print("  grande sólo suma ingeniería de proyecto — no señal adicional.")
