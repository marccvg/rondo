"""
Fase 3 — Optuna hyperparameter tuning + SHAP interpretability.

Sobre las 61 features de Fase 2 FULL, buscar mejores hiperparámetros
del LightGBM Poisson y explicar el modelo final vía SHAP.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dixon_coles import matrix_to_1x2, score_matrix
from elo import compute_elo_history
from fase1_baseline import home_prior_baseline, pinnacle_probs, uniform_baseline
from fase2_full_pipeline import merge_xg
from features.family_a_strength import compute_family_a
from features.family_b_form import compute_family_b
from features.family_c_context import compute_family_c
from features.family_e_tactical import compute_family_e
from features.family_f_matchup import compute_family_f
from metrics import all_metrics

ROOT = Path("/home/Projects/futbol_elo_poisson")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_dataset():
    """Reconstruye features FULL (mismo que Fase 2)."""
    matches = pd.read_parquet(PROC / "matches_all.parquet")
    understat = pd.read_parquet(ROOT / "data/raw/understat_full/understat_schedule_5big.parquet")
    df = merge_xg(matches, understat)
    df = compute_elo_history(df)
    fa = compute_family_a(df)
    fb = compute_family_b(df)
    fc = compute_family_c(df)
    ff = compute_family_f(df)
    df_xg = df.dropna(subset=["xg_home", "xg_away"]).copy()
    fe = compute_family_e(df_xg)
    features_no_e = pd.concat([fa, fb, fc, ff], axis=1)
    fe_full = pd.DataFrame(index=df.index, columns=fe.columns)
    fe_full.loc[df_xg.index] = fe.values
    all_feats = pd.concat([features_no_e, fe_full], axis=1)
    all_feats["target_goals_home"] = df["home_goals"].to_numpy()
    all_feats["target_goals_away"] = df["away_goals"].to_numpy()
    all_feats["target_result"] = df["result"].to_numpy()
    all_feats["meta_date"] = df["date"].to_numpy()
    all_feats["meta_season"] = df["season"].to_numpy()
    all_feats["meta_pinnacle_home"] = df["pinnacle_home"].to_numpy()
    all_feats["meta_pinnacle_draw"] = df["pinnacle_draw"].to_numpy()
    all_feats["meta_pinnacle_away"] = df["pinnacle_away"].to_numpy()
    return all_feats


def temporal_split(df):
    train_seasons = [f"20{y:02d}-{y+1:02d}" for y in range(17, 23)]
    train = df[df["meta_season"].isin(train_seasons)].copy()
    val = df[df["meta_season"] == "2023-24"].copy()
    test = df[df["meta_season"] == "2024-25"].copy()
    return train, val, test


def objective_maker(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            "objective": "poisson",
            "metric": "poisson",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True),
            "verbose": -1,
            "seed": 42,
        }
        train_ds = lgb.Dataset(X_train, y_train)
        val_ds = lgb.Dataset(X_val, y_val, reference=train_ds)
        model = lgb.train(
            params, train_ds, num_boost_round=2000,
            valid_sets=[val_ds], valid_names=["val"],
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
        )
        return model.best_score["val"]["poisson"]
    return objective


def probs_from_lambdas(lh, la, rho=-0.03):
    probs = np.zeros((len(lh), 3))
    for i in range(len(lh)):
        m = score_matrix(max(lh[i], 0.05), max(la[i], 0.05), rho=rho, max_goals=8)
        probs[i] = matrix_to_1x2(m)
    return probs


def main(n_trials: int = 40):
    print(">>> Fase 3 — Optuna + SHAP")
    print(f"Building dataset...")
    df = build_dataset()
    feature_cols = [c for c in df.columns if not c.startswith("target_") and not c.startswith("meta_")]
    print(f"Features: {len(feature_cols)}")

    train, val, test = temporal_split(df)
    print(f"Train {len(train):,} | Val {len(val):,} | Test {len(test):,}")

    Xt = train[feature_cols].astype(np.float32).to_numpy()
    Xv = val[feature_cols].astype(np.float32).to_numpy()
    Xtest = test[feature_cols].astype(np.float32).to_numpy()

    print(f"\n=== Optuna tuning ({n_trials} trials × 2 modelos) ===")
    print("Modelo HOME...")
    study_h = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_h.optimize(
        objective_maker(Xt, train["target_goals_home"].to_numpy(), Xv, val["target_goals_home"].to_numpy()),
        n_trials=n_trials, show_progress_bar=False,
    )
    print(f"Best val loss HOME: {study_h.best_value:.5f}")
    print(f"Best params: {study_h.best_params}")

    print("\nModelo AWAY...")
    study_a = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_a.optimize(
        objective_maker(Xt, train["target_goals_away"].to_numpy(), Xv, val["target_goals_away"].to_numpy()),
        n_trials=n_trials, show_progress_bar=False,
    )
    print(f"Best val loss AWAY: {study_a.best_value:.5f}")
    print(f"Best params: {study_a.best_params}")

    # Reentrenar con best params sobre train+val (más data), evaluar en test
    print("\n=== Reentreno final con best params ===")
    combined_X = np.vstack([Xt, Xv])
    combined_yh = np.concatenate([train["target_goals_home"].to_numpy(), val["target_goals_home"].to_numpy()])
    combined_ya = np.concatenate([train["target_goals_away"].to_numpy(), val["target_goals_away"].to_numpy()])

    params_h = {**study_h.best_params, "objective": "poisson", "metric": "poisson", "verbose": -1, "seed": 42}
    params_a = {**study_a.best_params, "objective": "poisson", "metric": "poisson", "verbose": -1, "seed": 42}

    # Usamos best_iter previo como num_boost_round aproximado
    print(f"Reentreno HOME con best params...")
    m_h = lgb.train(params_h, lgb.Dataset(combined_X, combined_yh), num_boost_round=500)
    print(f"Reentreno AWAY con best params...")
    m_a = lgb.train(params_a, lgb.Dataset(combined_X, combined_ya), num_boost_round=500)

    # Predict test
    test_lh = m_h.predict(Xtest)
    test_la = m_a.predict(Xtest)
    test_probs = probs_from_lambdas(test_lh, test_la, rho=-0.03)

    test_true = test["target_result"].to_numpy()
    test_pin = test.rename(columns={
        "meta_pinnacle_home": "pinnacle_home",
        "meta_pinnacle_draw": "pinnacle_draw",
        "meta_pinnacle_away": "pinnacle_away",
    })
    metrics = [
        all_metrics(test_true, uniform_baseline(len(test)), "Uniforme"),
        all_metrics(test_true, home_prior_baseline(len(test), train["target_result"].value_counts().to_dict()), "Home prior"),
        all_metrics(test_true, test_probs, "LGB Fase 3 (Optuna tuned)"),
        all_metrics(test_true, pinnacle_probs(test_pin), "Pinnacle"),
    ]
    print("\n=== TEST 2024-25 ===")
    for m in metrics:
        print(f"  {m['model']:30s}  log-loss={m['log_loss']:.4f}  RPS={m['rps']:.4f}  Brier={m['brier']:.4f}")

    # SHAP
    print("\n=== SHAP analysis (modelo HOME sobre test) ===")
    try:
        import shap
        explainer = shap.TreeExplainer(m_h)
        shap_values = explainer.shap_values(Xtest[:500])  # sample para velocidad
        # Aggregated importance
        shap_imp = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": feature_cols,
            "shap_importance": shap_imp,
            "lgb_gain": m_h.feature_importance("gain"),
        }).sort_values("shap_importance", ascending=False)
        print("\nTop 15 features por SHAP importance (modelo HOME):")
        print(shap_df.head(15).to_string(index=False))
    except Exception as e:
        print(f"SHAP falló: {e}")
        shap_df = None

    # Guardar modelos
    m_h.save_model(str(MODELS / "lgb_fase3_home.txt"))
    m_a.save_model(str(MODELS / "lgb_fase3_away.txt"))
    print(f"\nModelos guardados en {MODELS}")

    # Reporte
    report = REPORTS / "fase3_optuna.md"
    with open(report, "w") as f:
        f.write("# Fase 3 — Optuna tuning + SHAP\n\n")
        f.write(f"Generado: {pd.Timestamp.now().isoformat()}\n\n")
        f.write(f"## Setup\n\n- Features: {len(feature_cols)}\n")
        f.write(f"- Optuna trials: {n_trials} × 2 modelos = {n_trials*2}\n")
        f.write(f"- Reentrenamiento final sobre train+val combinados\n\n")
        f.write(f"## Best hyperparams\n\n**HOME model** (val loss {study_h.best_value:.5f}):\n```\n{study_h.best_params}\n```\n\n")
        f.write(f"**AWAY model** (val loss {study_a.best_value:.5f}):\n```\n{study_a.best_params}\n```\n\n")
        f.write("## Métricas TEST 2024-25\n\n| Modelo | log-loss | RPS | Brier |\n|---|---|---|---|\n")
        for m in metrics:
            f.write(f"| {m['model']} | {m['log_loss']:.4f} | {m['rps']:.4f} | {m['brier']:.4f} |\n")
        model_ll = next(m['log_loss'] for m in metrics if 'Fase 3' in m['model'])
        pin_ll = next(m['log_loss'] for m in metrics if m['model'] == 'Pinnacle')
        f.write(f"\n## Progreso vs fases anteriores\n\n")
        f.write(f"| Fase | log-loss test | Ratio vs Pinnacle |\n|---|---|---|\n")
        f.write(f"| 1 baseline | 0.9859 | 1.026 |\n")
        f.write(f"| 2 básica (sin xG) | 0.9888 | 1.029 |\n")
        f.write(f"| 2 FULL (con xG) | 0.9818 | 1.022 |\n")
        f.write(f"| **3 Optuna tuned** | **{model_ll:.4f}** | **{model_ll/pin_ll:.4f}** |\n\n")
        if shap_df is not None:
            f.write("## SHAP top 15 features (modelo HOME)\n\n")
            f.write(shap_df.head(15).to_markdown(index=False))
    print(f"\nReporte: {report}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=40)
    args = p.parse_args()
    main(n_trials=args.trials)
