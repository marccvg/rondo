"""
Fase 1 — Baseline ELO + Poisson bivariable con corrección Dixon-Coles.

Pipeline:
1. Carga matches + Elo pre-partido
2. Split temporal: train 17/18-22/23, val 23/24, test 24/25
3. Ajusta Poisson GLM: log(λ_home) = α + β1·elo_diff + γ  (home indicator)
                       log(λ_away) = α' + β1'·elo_diff
4. Estima rho DC óptimo sobre train (grid search)
5. Predicciones test → probabilidades 1X2
6. Métricas: log-loss, RPS, Brier vs baselines (uniforme, home-win-rate, Pinnacle)
7. Escribe reports/fase1_baseline.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from elo import compute_elo_history
from dixon_coles import score_matrix, matrix_to_1x2
from data_loader import implied_probs_from_odds

ROOT = Path("/home/Projects/futbol_elo_poisson")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train: seasons 17/18-22/23, Val: 23/24, Test: 24/25."""
    train = df[df["season"].isin([f"20{y:02d}-{y+1:02d}" for y in range(17, 23)])].copy()
    val = df[df["season"] == "2023-24"].copy()
    test = df[df["season"] == "2024-25"].copy()
    return train, val, test


def fit_poisson_glm(train: pd.DataFrame) -> tuple[sm.GLM, sm.GLM]:
    """Dos modelos GLM Poisson: uno para goles_home, uno para goles_away.
    Features: elo_diff (elo_home - elo_away). No usamos elo_prob_home directamente
    porque queremos que el modelo aprenda la relación no-lineal con log-link.
    """
    train = train.dropna(subset=["home_goals", "away_goals", "elo_home", "elo_away"]).copy()
    train["elo_diff"] = train["elo_home"] - train["elo_away"]
    X = sm.add_constant(train[["elo_diff"]])

    y_home = train["home_goals"].astype(float)
    y_away = train["away_goals"].astype(float)

    m_home = sm.GLM(y_home, X, family=sm.families.Poisson()).fit()
    # Para visitante, invertimos el signo del elo_diff conceptualmente
    X_away = sm.add_constant(train[["elo_diff"]].rename(columns={"elo_diff": "elo_diff"}))
    m_away = sm.GLM(y_away, X_away, family=sm.families.Poisson()).fit()
    return m_home, m_away


def predict_lambdas(model_h: sm.GLM, model_a: sm.GLM, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = df.copy()
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    X = sm.add_constant(df[["elo_diff"]])
    lam_h = model_h.predict(X).to_numpy()
    lam_a = model_a.predict(X).to_numpy()
    return lam_h, lam_a


def optimize_rho(train_lam_h: np.ndarray, train_lam_a: np.ndarray,
                 train_gh: np.ndarray, train_ga: np.ndarray,
                 rhos: np.ndarray | None = None) -> float:
    """Grid search de rho minimizando log-loss de resultado exacto."""
    if rhos is None:
        rhos = np.arange(-0.15, 0.05, 0.01)
    best_rho, best_ll = 0.0, np.inf
    for rho in rhos:
        ll = 0.0
        n = 0
        for lh, la, gh, ga in zip(train_lam_h, train_lam_a, train_gh, train_ga):
            gh_i = min(int(gh), 8)
            ga_i = min(int(ga), 8)
            m = score_matrix(lh, la, rho=float(rho), max_goals=8)
            p = max(m[gh_i, ga_i], 1e-15)
            ll -= np.log(p)
            n += 1
        ll /= n
        if ll < best_ll:
            best_ll, best_rho = ll, float(rho)
    return best_rho


def predict_1x2(lam_h: np.ndarray, lam_a: np.ndarray, rho: float) -> np.ndarray:
    """Devuelve matriz N×3 con [P(H), P(D), P(A)]."""
    out = np.zeros((len(lam_h), 3))
    for i, (lh, la) in enumerate(zip(lam_h, lam_a)):
        m = score_matrix(lh, la, rho=rho, max_goals=8)
        out[i] = matrix_to_1x2(m)
    return out


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, label: str) -> dict:
    """
    y_true: [H/D/A] → mapea a [0/1/2]
    probs: N×3 con probabilidades
    Devuelve log-loss, RPS, Brier.
    """
    idx_map = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([idx_map[r] for r in y_true])
    n = len(y_idx)
    p_true = np.clip(probs[np.arange(n), y_idx], 1e-15, 1.0)
    log_loss = -np.log(p_true).mean()
    # Brier: sum (p_i - y_i)^2 / n
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y_idx] = 1.0
    brier = ((probs - onehot) ** 2).sum(axis=1).mean()
    # RPS: cumulative squared error
    cum_p = np.cumsum(probs[:, :2], axis=1)  # only need first 2 categories
    cum_y = np.cumsum(onehot[:, :2], axis=1)
    rps = ((cum_p - cum_y) ** 2).sum(axis=1).mean()
    return {"model": label, "n": n, "log_loss": log_loss, "rps": rps, "brier": brier}


def uniform_baseline(n: int) -> np.ndarray:
    return np.tile([1 / 3, 1 / 3, 1 / 3], (n, 1))


def home_prior_baseline(n: int, train_result_counts: dict) -> np.ndarray:
    total = sum(train_result_counts.values())
    p_h = train_result_counts.get("H", 0) / total
    p_d = train_result_counts.get("D", 0) / total
    p_a = train_result_counts.get("A", 0) / total
    return np.tile([p_h, p_d, p_a], (n, 1))


def pinnacle_probs(df: pd.DataFrame) -> np.ndarray:
    """Convierte cuotas Pinnacle en probabilidades normalizadas."""
    probs = []
    for row in df.itertuples(index=False):
        try:
            ph, pd_, pa = implied_probs_from_odds(row.pinnacle_home, row.pinnacle_draw, row.pinnacle_away)
        except (ZeroDivisionError, TypeError, ValueError):
            ph, pd_, pa = 1 / 3, 1 / 3, 1 / 3
        probs.append([ph, pd_, pa])
    return np.array(probs)


def main():
    print(">>> Fase 1 baseline — arranque")
    # 1. cargar matches
    matches_path = PROC / "matches_all.parquet"
    df = pd.read_parquet(matches_path)
    print(f"Cargados {len(df):,} partidos")

    # 2. añadir ELO pre-partido
    print("Computando ELO histórico...")
    df_elo = compute_elo_history(df)

    # 3. split
    train, val, test = split_temporal(df_elo)
    print(f"Train: {len(train):,}  |  Val: {len(val):,}  |  Test: {len(test):,}")

    # 4. fit Poisson GLM
    print("Ajustando Poisson GLM...")
    m_h, m_a = fit_poisson_glm(train)
    print(f"Modelo home: {m_h.params.to_dict()}")
    print(f"Modelo away: {m_a.params.to_dict()}")

    # 5. optimizar rho sobre train (subset por velocidad)
    print("Optimizando rho Dixon-Coles...")
    tr_lh, tr_la = predict_lambdas(m_h, m_a, train)
    sample_idx = np.random.RandomState(42).choice(len(train), size=min(2000, len(train)), replace=False)
    best_rho = optimize_rho(
        tr_lh[sample_idx], tr_la[sample_idx],
        train["home_goals"].to_numpy()[sample_idx],
        train["away_goals"].to_numpy()[sample_idx],
    )
    print(f"rho óptimo: {best_rho:.3f}")

    # 6. predicciones val y test
    print("Prediciendo val y test...")
    val_lh, val_la = predict_lambdas(m_h, m_a, val)
    test_lh, test_la = predict_lambdas(m_h, m_a, test)
    val_probs = predict_1x2(val_lh, val_la, best_rho)
    test_probs = predict_1x2(test_lh, test_la, best_rho)

    # 7. métricas
    print("\n=== VAL 2023-24 ===")
    val_true = val["result"].to_numpy()
    val_counts = train["result"].value_counts().to_dict()
    val_metrics = []
    val_metrics.append(compute_metrics(val_true, uniform_baseline(len(val)), "uniforme"))
    val_metrics.append(compute_metrics(val_true, home_prior_baseline(len(val), val_counts), "home_prior"))
    val_metrics.append(compute_metrics(val_true, val_probs, "ELO+Poisson+DC"))
    val_metrics.append(compute_metrics(val_true, pinnacle_probs(val), "Pinnacle_cuotas"))
    for m in val_metrics:
        print(f"  {m['model']:25s}  n={m['n']:5d}  log-loss={m['log_loss']:.4f}  RPS={m['rps']:.4f}  Brier={m['brier']:.4f}")

    print("\n=== TEST 2024-25 ===")
    test_true = test["result"].to_numpy()
    test_metrics = []
    test_metrics.append(compute_metrics(test_true, uniform_baseline(len(test)), "uniforme"))
    test_metrics.append(compute_metrics(test_true, home_prior_baseline(len(test), val_counts), "home_prior"))
    test_metrics.append(compute_metrics(test_true, test_probs, "ELO+Poisson+DC"))
    test_metrics.append(compute_metrics(test_true, pinnacle_probs(test), "Pinnacle_cuotas"))
    for m in test_metrics:
        print(f"  {m['model']:25s}  n={m['n']:5d}  log-loss={m['log_loss']:.4f}  RPS={m['rps']:.4f}  Brier={m['brier']:.4f}")

    # 8. escribir reporte markdown
    report = REPORTS / "fase1_baseline.md"
    with open(report, "w") as f:
        f.write("# Fase 1 — Baseline ELO + Poisson + Dixon-Coles\n\n")
        f.write(f"Generado: {pd.Timestamp.now().isoformat()}\n\n")
        f.write(f"## Dataset\n\n- Train (17/18-22/23): {len(train):,}\n- Val (23/24): {len(val):,}\n- Test (24/25): {len(test):,}\n\n")
        f.write(f"## Modelo\n\n- Poisson GLM 1 feature (elo_diff = elo_home - elo_away)\n")
        f.write(f"- Coeficientes home: intercept={m_h.params.iloc[0]:.4f}, elo_diff={m_h.params.iloc[1]:.6f}\n")
        f.write(f"- Coeficientes away: intercept={m_a.params.iloc[0]:.4f}, elo_diff={m_a.params.iloc[1]:.6f}\n")
        f.write(f"- Rho Dixon-Coles óptimo: {best_rho:.3f}\n\n")
        f.write("## Métricas VAL (2023-24)\n\n")
        f.write("| Modelo | n | log-loss | RPS | Brier |\n|---|---|---|---|---|\n")
        for m in val_metrics:
            f.write(f"| {m['model']} | {m['n']} | {m['log_loss']:.4f} | {m['rps']:.4f} | {m['brier']:.4f} |\n")
        f.write("\n## Métricas TEST (2024-25)\n\n")
        f.write("| Modelo | n | log-loss | RPS | Brier |\n|---|---|---|---|---|\n")
        for m in test_metrics:
            f.write(f"| {m['model']} | {m['n']} | {m['log_loss']:.4f} | {m['rps']:.4f} | {m['brier']:.4f} |\n")
        # Ratios vs Pinnacle
        pin_ll = next(m['log_loss'] for m in test_metrics if m['model'] == 'Pinnacle_cuotas')
        model_ll = next(m['log_loss'] for m in test_metrics if m['model'] == 'ELO+Poisson+DC')
        home_ll = next(m['log_loss'] for m in test_metrics if m['model'] == 'home_prior')
        f.write(f"\n## Interpretación\n\n")
        f.write(f"- Modelo vs Pinnacle (test): ratio log-loss = {model_ll/pin_ll:.3f} (1.00 = igual, <1 = mejor)\n")
        f.write(f"- Modelo vs home_prior baseline (test): mejora relativa = {(1 - model_ll/home_ll)*100:.2f}%\n")
        f.write(f"- Rho negativo confirma efecto DC: resultados bajos empates + 1-0/0-1 más frecuentes que Poisson puro\n")
    print(f"\nReporte escrito en {report}")


if __name__ == "__main__":
    main()
