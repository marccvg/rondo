"""
Fase 2 (parcial) — LightGBM Poisson sobre features básicos (familias A+B+C+F).

Objetivo: ver cuánto mejora frente al baseline ELO+Poisson+DC (Fase 1) SIN aún haber
integrado features FBref (xG, presiones, pases progresivos). Si esto ya supera baseline
por >3%, señal de que el enfoque escala. Si no, indica que el salto grande sólo llega
con features tácticos avanzados.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dixon_coles import matrix_to_1x2, score_matrix
from fase1_baseline import (
    home_prior_baseline,
    pinnacle_probs,
    uniform_baseline,
)
from metrics import all_metrics

ROOT = Path("/home/Projects/futbol_elo_poisson")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def split_temporal(df: pd.DataFrame):
    train = df[df["meta_season"].isin([f"20{y:02d}-{y+1:02d}" for y in range(17, 23)])].copy()
    val = df[df["meta_season"] == "2023-24"].copy()
    test = df[df["meta_season"] == "2024-25"].copy()
    return train, val, test


def train_lgb_poisson(X_train, y_train, X_val, y_val, seed: int = 42):
    """LightGBM con loss Poisson y early stopping sobre val."""
    train_ds = lgb.Dataset(X_train, y_train)
    val_ds = lgb.Dataset(X_val, y_val, reference=train_ds)
    params = {
        "objective": "poisson",
        "metric": "poisson",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": seed,
    }
    model = lgb.train(
        params, train_ds, num_boost_round=1500,
        valid_sets=[val_ds], valid_names=["val"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def probs_from_lambdas(lam_h: np.ndarray, lam_a: np.ndarray, rho: float = -0.03) -> np.ndarray:
    probs = np.zeros((len(lam_h), 3))
    for i, (lh, la) in enumerate(zip(lam_h, lam_a)):
        m = score_matrix(max(lh, 0.05), max(la, 0.05), rho=rho, max_goals=8)
        probs[i] = matrix_to_1x2(m)
    return probs


def main():
    print(">>> Fase 2 LightGBM (features básicos A+B+C+F) — arranque")
    df = pd.read_parquet(PROC / "features_basic.parquet")
    print(f"Base features: {len(df):,} filas × {len(df.columns)} cols")

    feature_cols = [c for c in df.columns if not c.startswith("target_") and not c.startswith("meta_")]
    print(f"Features a usar: {len(feature_cols)}")

    train, val, test = split_temporal(df)
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    # Drop filas con muchos NaN en features iniciales (primeros partidos sin histórico)
    def clean(sub):
        # LightGBM tolera NaN, no filtro
        return sub

    Xt = train[feature_cols].to_numpy(dtype=np.float32)
    Xv = val[feature_cols].to_numpy(dtype=np.float32)
    Xtest = test[feature_cols].to_numpy(dtype=np.float32)

    print("\nEntrenando modelo goles_home...")
    m_h = train_lgb_poisson(
        Xt, train["target_goals_home"].to_numpy(),
        Xv, val["target_goals_home"].to_numpy(),
    )
    print(f"Best iter home: {m_h.best_iteration}")

    print("Entrenando modelo goles_away...")
    m_a = train_lgb_poisson(
        Xt, train["target_goals_away"].to_numpy(),
        Xv, val["target_goals_away"].to_numpy(),
    )
    print(f"Best iter away: {m_a.best_iteration}")

    # Predicciones
    val_lh = m_h.predict(Xv, num_iteration=m_h.best_iteration)
    val_la = m_a.predict(Xv, num_iteration=m_a.best_iteration)
    test_lh = m_h.predict(Xtest, num_iteration=m_h.best_iteration)
    test_la = m_a.predict(Xtest, num_iteration=m_a.best_iteration)

    # rho DC — usamos el mismo optimizado en Fase 1 para comparación limpia
    rho = -0.030
    val_probs = probs_from_lambdas(val_lh, val_la, rho=rho)
    test_probs = probs_from_lambdas(test_lh, test_la, rho=rho)

    print("\n=== VAL 2023-24 ===")
    val_true = val["target_result"].to_numpy()
    val_counts = train["target_result"].value_counts().to_dict()
    val_metrics = [
        all_metrics(val_true, uniform_baseline(len(val)), "uniforme"),
        all_metrics(val_true, home_prior_baseline(len(val), val_counts), "home_prior"),
        all_metrics(val_true, val_probs, "LightGBM(A+B+C+F)"),
        all_metrics(val_true, pinnacle_probs(val.rename(columns={
            "meta_pinnacle_home": "pinnacle_home",
            "meta_pinnacle_draw": "pinnacle_draw",
            "meta_pinnacle_away": "pinnacle_away",
        })), "Pinnacle_cuotas"),
    ]
    for m in val_metrics:
        print(f"  {m['model']:22s}  n={m['n']:5d}  log-loss={m['log_loss']:.4f}  RPS={m['rps']:.4f}  Brier={m['brier']:.4f}")

    print("\n=== TEST 2024-25 ===")
    test_true = test["target_result"].to_numpy()
    test_metrics = [
        all_metrics(test_true, uniform_baseline(len(test)), "uniforme"),
        all_metrics(test_true, home_prior_baseline(len(test), val_counts), "home_prior"),
        all_metrics(test_true, test_probs, "LightGBM(A+B+C+F)"),
        all_metrics(test_true, pinnacle_probs(test.rename(columns={
            "meta_pinnacle_home": "pinnacle_home",
            "meta_pinnacle_draw": "pinnacle_draw",
            "meta_pinnacle_away": "pinnacle_away",
        })), "Pinnacle_cuotas"),
    ]
    for m in test_metrics:
        print(f"  {m['model']:22s}  n={m['n']:5d}  log-loss={m['log_loss']:.4f}  RPS={m['rps']:.4f}  Brier={m['brier']:.4f}")

    # Feature importance top 15
    imp = pd.DataFrame({
        "feature": feature_cols,
        "imp_home": m_h.feature_importance(importance_type="gain"),
        "imp_away": m_a.feature_importance(importance_type="gain"),
    })
    imp["imp_total"] = imp["imp_home"] + imp["imp_away"]
    imp_top = imp.sort_values("imp_total", ascending=False).head(15)
    print("\n=== Top 15 features (importance by gain) ===")
    print(imp_top.to_string(index=False))

    # Reporte
    report = REPORTS / "fase2_lgb_basic.md"
    with open(report, "w") as f:
        f.write("# Fase 2 (parcial) — LightGBM sobre features básicos (A+B+C+F, sin FBref)\n\n")
        f.write(f"Generado: {pd.Timestamp.now().isoformat()}\n\n")
        f.write(f"## Setup\n\n- Features: {len(feature_cols)} ({', '.join(feature_cols[:8])}, ...)\n")
        f.write(f"- LightGBM Poisson, learning_rate=0.05, num_leaves=31, min_data_in_leaf=30\n")
        f.write(f"- Best iter home: {m_h.best_iteration}, away: {m_a.best_iteration}\n")
        f.write(f"- rho Dixon-Coles: {rho}\n\n")
        f.write("## Métricas TEST (2024-25)\n\n")
        f.write("| Modelo | log-loss | RPS | Brier |\n|---|---|---|---|\n")
        for m in test_metrics:
            f.write(f"| {m['model']} | {m['log_loss']:.4f} | {m['rps']:.4f} | {m['brier']:.4f} |\n")
        model_ll = next(m['log_loss'] for m in test_metrics if 'LightGBM' in m['model'])
        pin_ll = next(m['log_loss'] for m in test_metrics if 'Pinnacle' in m['model'])
        # Compare vs Fase 1 baseline (0.9859)
        f.write(f"\n## Comparación vs Fase 1 baseline (0.9859)\n\n")
        delta = (0.9859 - model_ll) / 0.9859 * 100
        f.write(f"- Fase 2 log-loss test: **{model_ll:.4f}**\n")
        f.write(f"- Mejora relativa vs Fase 1: **{delta:+.2f}%**\n")
        f.write(f"- Ratio vs Pinnacle: **{model_ll/pin_ll:.3f}** (Fase 1 fue 1.026)\n\n")
        f.write("## Top 15 features importancia\n\n")
        f.write(imp_top.to_markdown(index=False))
    print(f"\nReporte: {report}")


if __name__ == "__main__":
    main()
