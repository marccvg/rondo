"""
Entrena LightGBM predictor puntos FPL próxima jornada.
Split temporal: train seasons 19/20-24/25, test season 25/26.
"""
from __future__ import annotations
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path("/home/Projects/futbol_elo_poisson/fantasy")
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def train():
    df = pd.read_parquet(PROC / "fpl_features.parquet")
    print(f"Base: {len(df):,} rows")

    # Features
    feature_cols = [c for c in df.columns if any(
        c.startswith(prefix) for prefix in [
            "total_points_r", "minutes_r", "goals_scored_r", "assists_r",
            "clean_sheets_r", "saves_r", "bonus_r", "xP_r",
            "expected_goals_r", "expected_assists_r", "starts_r"
        ])]
    feature_cols += ["form_fpl", "value_M", "ppm_r5", "trend_points", "was_home"]
    # Encoding posición
    df = pd.get_dummies(df, columns=["position"], prefix="pos", dummy_na=False)
    feature_cols += [c for c in df.columns if c.startswith("pos_")]

    print(f"Features usadas: {len(feature_cols)}")

    # Filtrar rows con target válido y minutos > 0 en próximo (evitar predecir puntos de no-jugadores)
    df = df.dropna(subset=["target_next_gw_points"])
    print(f"Con target: {len(df):,}")

    # Split temporal
    train = df[df["_season"].isin(["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"])].copy()
    val = df[df["_season"] == "2024-25"].copy()
    test = df[df["_season"] == "2025-26"].copy()
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    Xt = train[feature_cols].astype(np.float32).values
    Xv = val[feature_cols].astype(np.float32).values
    Xtest = test[feature_cols].astype(np.float32).values

    y_t = train["target_next_gw_points"].values
    y_v = val["target_next_gw_points"].values
    y_test = test["target_next_gw_points"].values

    # LightGBM regressor
    params = {
        "objective": "regression",
        "metric": ["mae", "rmse"],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l2": 0.5,
        "verbose": -1,
        "seed": 42,
    }
    train_ds = lgb.Dataset(Xt, y_t, feature_name=feature_cols)
    val_ds = lgb.Dataset(Xv, y_v, reference=train_ds)

    print("\nEntrenando LightGBM...")
    model = lgb.train(
        params, train_ds, num_boost_round=1500,
        valid_sets=[val_ds], valid_names=["val"],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"Best iter: {model.best_iteration}")

    # Predicciones test
    pred_test = model.predict(Xtest, num_iteration=model.best_iteration)
    mae = np.mean(np.abs(pred_test - y_test))
    rmse = np.sqrt(np.mean((pred_test - y_test) ** 2))

    # Baselines para comparación
    baseline_mean = np.full_like(y_test, y_t.mean(), dtype=float)
    baseline_form = test["form_fpl"].fillna(y_t.mean()).values
    baseline_xP = test["xP"].fillna(y_t.mean()).values if "xP" in test.columns else baseline_form

    mae_mean = np.mean(np.abs(baseline_mean - y_test))
    mae_form = np.mean(np.abs(baseline_form - y_test))
    mae_xP = np.mean(np.abs(baseline_xP - y_test))

    print(f"\n=== MÉTRICAS TEST 25/26 ===")
    print(f"  MAE modelo LGB:        {mae:.3f}")
    print(f"  MAE baseline mean:     {mae_mean:.3f}")
    print(f"  MAE baseline form_fpl: {mae_form:.3f}")
    print(f"  MAE baseline xP FPL:   {mae_xP:.3f}  ← es el que da la propia app FPL")
    print(f"  RMSE modelo LGB:       {rmse:.3f}")

    # Feature importance
    imp = pd.DataFrame({
        "feature": feature_cols,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    print(f"\nTop 15 features por gain:")
    print(imp.head(15).to_string(index=False))

    # Guardar
    model.save_model(str(MODELS / "fpl_predictor_lgb.txt"))
    imp.to_csv(MODELS / "feature_importance.csv", index=False)
    with open(MODELS / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    print(f"\n✅ Modelo guardado: {MODELS / 'fpl_predictor_lgb.txt'}")


if __name__ == "__main__":
    train()
