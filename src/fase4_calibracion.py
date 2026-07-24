"""
Fase 4 — Calibración probabilística.

Objetivo: verificar que cuando predecimos P(H)=40% realmente gana el local ~40%.
Si no, aplicar corrección post-hoc (isotonic regression o Platt scaling).

Genera reliability diagrams por clase (H, D, A) y calibra si mejora log-loss.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))
from dixon_coles import matrix_to_1x2, score_matrix
from fase1_baseline import pinnacle_probs
from fase3_optuna_shap import build_dataset, temporal_split
from metrics import all_metrics

ROOT = Path("/home/Projects/futbol_elo_poisson")
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


def probs_from_lambdas(lh, la, rho=-0.03):
    probs = np.zeros((len(lh), 3))
    for i in range(len(lh)):
        m = score_matrix(max(lh[i], 0.05), max(la[i], 0.05), rho=rho, max_goals=8)
        probs[i] = matrix_to_1x2(m)
    return probs


def reliability_table(y_true, probs_class, class_label, n_bins=10):
    """Para una clase (H, D o A), agrupa predicciones en buckets y compara con freq real."""
    idx_map = {"H": 0, "D": 1, "A": 2}
    class_idx = idx_map[class_label]
    y_binary = (y_true == class_label).astype(int)
    p_class = probs_class[:, class_idx]

    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p_class >= lo) & (p_class < hi) if i < n_bins-1 else (p_class >= lo) & (p_class <= hi)
        n = mask.sum()
        if n == 0:
            continue
        pred_mean = p_class[mask].mean()
        actual_freq = y_binary[mask].mean()
        rows.append({
            "bin": f"[{lo:.2f},{hi:.2f})",
            "n": int(n),
            "predicted_mean": pred_mean,
            "actual_freq": actual_freq,
            "gap": actual_freq - pred_mean,
        })
    return pd.DataFrame(rows)


def main():
    print(">>> Fase 4 — Calibración")
    df = build_dataset()
    feature_cols = [c for c in df.columns if not c.startswith("target_") and not c.startswith("meta_")]
    train, val, test = temporal_split(df)

    Xt = train[feature_cols].astype(np.float32).to_numpy()
    Xv = val[feature_cols].astype(np.float32).to_numpy()
    Xtest = test[feature_cols].astype(np.float32).to_numpy()

    # Cargar modelos Fase 3
    m_h = lgb.Booster(model_file=str(MODELS / "lgb_fase3_home.txt"))
    m_a = lgb.Booster(model_file=str(MODELS / "lgb_fase3_away.txt"))

    val_lh = m_h.predict(Xv)
    val_la = m_a.predict(Xv)
    test_lh = m_h.predict(Xtest)
    test_la = m_a.predict(Xtest)

    val_probs = probs_from_lambdas(val_lh, val_la)
    test_probs = probs_from_lambdas(test_lh, test_la)

    val_true = val["target_result"].to_numpy()
    test_true = test["target_result"].to_numpy()

    print("\n=== Reliability diagrams (TEST 2024-25, sin calibrar) ===")
    for cls in ["H", "D", "A"]:
        print(f"\nClase {cls}:")
        rt = reliability_table(test_true, test_probs, cls)
        print(rt.to_string(index=False))
        max_gap = rt["gap"].abs().max()
        print(f"  Max |gap| = {max_gap:.4f}")

    # Calibración isotonic por clase, ajustada sobre val
    print("\n=== Aplicando isotonic regression (fit val, apply test) ===")
    calibrated_test = np.zeros_like(test_probs)
    for cls_idx, cls in enumerate(["H", "D", "A"]):
        iso = IsotonicRegression(out_of_bounds="clip")
        y_val_bin = (val_true == cls).astype(int)
        iso.fit(val_probs[:, cls_idx], y_val_bin)
        calibrated_test[:, cls_idx] = iso.predict(test_probs[:, cls_idx])

    # Renormalizar por fila
    calibrated_test = calibrated_test / calibrated_test.sum(axis=1, keepdims=True)

    print("\n=== Reliability POST-calibración ===")
    for cls in ["H", "D", "A"]:
        rt = reliability_table(test_true, calibrated_test, cls)
        max_gap = rt["gap"].abs().max()
        print(f"Clase {cls}: max |gap| = {max_gap:.4f}")

    # Métricas comparadas
    print("\n=== Métricas ANTES vs DESPUÉS de calibrar ===")
    pin_test = test.rename(columns={
        "meta_pinnacle_home": "pinnacle_home",
        "meta_pinnacle_draw": "pinnacle_draw",
        "meta_pinnacle_away": "pinnacle_away",
    })
    for name, probs in [
        ("Sin calibrar", test_probs),
        ("Con isotonic", calibrated_test),
        ("Pinnacle", pinnacle_probs(pin_test)),
    ]:
        m = all_metrics(test_true, probs, name)
        print(f"  {m['model']:20s}  log-loss={m['log_loss']:.4f}  RPS={m['rps']:.4f}  Brier={m['brier']:.4f}")

    # Guardar probs calibradas para dashboard
    np.save(MODELS / "test_probs_calibrated.npy", calibrated_test)
    print(f"\nProbs calibradas guardadas.")

    # Reporte
    report = REPORTS / "fase4_calibracion.md"
    with open(report, "w") as f:
        f.write("# Fase 4 — Calibración probabilística\n\n")
        f.write(f"Generado: {pd.Timestamp.now().isoformat()}\n\n")
        f.write("## Método\n\n- Isotonic regression aplicada por clase (H, D, A).\n")
        f.write("- Fit sobre val 2023/24, evaluación en test 2024/25.\n\n")
        f.write("## Reliability por clase (test)\n\n")
        for cls in ["H", "D", "A"]:
            rt_pre = reliability_table(test_true, test_probs, cls)
            rt_post = reliability_table(test_true, calibrated_test, cls)
            f.write(f"### Clase {cls}\n\n")
            f.write("**Pre-calibración**:\n\n")
            f.write(rt_pre.to_markdown(index=False))
            f.write(f"\n\n Max |gap| = {rt_pre['gap'].abs().max():.4f}\n\n")
            f.write("**Post-calibración isotonic**:\n\n")
            f.write(rt_post.to_markdown(index=False))
            f.write(f"\n\n Max |gap| = {rt_post['gap'].abs().max():.4f}\n\n")

        f.write("## Métricas comparadas\n\n| Modelo | log-loss | RPS | Brier |\n|---|---|---|---|\n")
        for name, probs in [
            ("Sin calibrar", test_probs),
            ("Con isotonic", calibrated_test),
            ("Pinnacle", pinnacle_probs(pin_test)),
        ]:
            m = all_metrics(test_true, probs, name)
            f.write(f"| {m['model']} | {m['log_loss']:.4f} | {m['rps']:.4f} | {m['brier']:.4f} |\n")
    print(f"\nReporte: {report}")


if __name__ == "__main__":
    main()
