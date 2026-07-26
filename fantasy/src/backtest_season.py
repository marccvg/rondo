"""
Backtest fantasy season 25/26 completo.

Simula:
  - Entrenar modelo SOLO con temporadas 19/20 - 24/25 (nunca ha visto 25/26)
  - Para cada GW de 25/26, predecir puntos y elegir plantilla óptima
  - Sumar puntos reales que HABRÍAMOS obtenido con esa plantilla
  - Comparar vs baselines:
    * Plantilla "top ownership" (los que la mayoría de gente elige)
    * Plantilla "top form" (los que están en mejor forma)
    * Plantilla "random"
    * Baseline xP oficial FPL
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import lightgbm as lgb
import numpy as np
import pandas as pd
import pulp

ROOT = Path("/home/Projects/futbol_elo_poisson/fantasy")
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

TRAIN_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
TEST_SEASON = "2025-26"


def train_model_no_test(df, feature_cols):
    train = df[df["_season"].isin(TRAIN_SEASONS)].dropna(subset=["target_next_gw_points"])
    Xt = train[feature_cols].astype(np.float32).values
    yt = train["target_next_gw_points"].values

    params = {
        "objective": "regression", "metric": "mae",
        "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
        "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
        "lambda_l2": 0.5, "verbose": -1, "seed": 42,
    }
    train_ds = lgb.Dataset(Xt, yt)
    print(f"  Entrenando sobre {len(train):,} rows (sin 25/26)...")
    model = lgb.train(params, train_ds, num_boost_round=100)
    return model


def optimize_xi_for_gw(pred_df, budget=100.0):
    """LP para elegir 11 titulares dado presupuesto (relajación 11 vs 15)."""
    d = pred_df.reset_index(drop=True)
    n = len(d)
    if n == 0:
        return []
    prob = pulp.LpProblem("fpl_xi", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    prob += pulp.lpSum(x[i] * d.loc[i, "pred_pts"] for i in range(n))
    prob += pulp.lpSum(x[i] * d.loc[i, "value"] for i in range(n)) <= budget * 10
    prob += pulp.lpSum(x[i] for i in range(n)) == 11
    # Formación válida: 1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD
    prob += pulp.lpSum(x[i] for i in range(n) if d.loc[i, "position"] == "GK") == 1
    prob += pulp.lpSum(x[i] for i in range(n) if d.loc[i, "position"] == "DEF") >= 3
    prob += pulp.lpSum(x[i] for i in range(n) if d.loc[i, "position"] == "DEF") <= 5
    prob += pulp.lpSum(x[i] for i in range(n) if d.loc[i, "position"] == "MID") >= 3
    prob += pulp.lpSum(x[i] for i in range(n) if d.loc[i, "position"] == "MID") <= 5
    prob += pulp.lpSum(x[i] for i in range(n) if d.loc[i, "position"] == "FWD") >= 1
    prob += pulp.lpSum(x[i] for i in range(n) if d.loc[i, "position"] == "FWD") <= 3
    # Max 3 por club
    for team in d["team"].unique():
        prob += pulp.lpSum(x[i] for i in range(n) if d.loc[i, "team"] == team) <= 3

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))
    selected = [i for i in range(n) if x[i].value() == 1]
    return d.loc[selected]


def main():
    print(">>> Backtest fantasy season 2025-26")
    df = pd.read_parquet(PROC / "fpl_features.parquet")
    print(f"Base features: {len(df):,} rows")

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

    # Re-add position column como texto para constraints LP
    df["position"] = np.select(
        [df.get("pos_GK", 0) == 1, df.get("pos_DEF", 0) == 1,
         df.get("pos_MID", 0) == 1, df.get("pos_FWD", 0) == 1],
        ["GK", "DEF", "MID", "FWD"],
        default="UNK"
    )

    print("Entrenando modelo (sin datos de 25/26)...")
    model = train_model_no_test(df, feature_cols)

    # Test set: solo 25/26
    test = df[df["_season"] == TEST_SEASON].copy()
    print(f"Test 25/26: {len(test):,} rows, {test['GW'].nunique()} GWs")

    # Predecir puntos próxima GW para cada row
    Xtest = test[feature_cols].astype(np.float32).values
    test["pred_pts"] = np.clip(model.predict(Xtest), 0, None)

    # Simular GW por GW
    print("\nSimulando fantasy GW por GW...")
    results = []
    all_gws = sorted(test["GW"].unique())

    for gw in all_gws:
        gw_data = test[test["GW"] == gw].copy()
        # Filtro: solo jugadores con minutos>0 esa GW (los que jugaron)
        # Para la predicción usamos pred_pts, pero puntos REALES son 'total_points' de esa row
        gw_data = gw_data[gw_data["value"] > 0]
        if len(gw_data) == 0:
            continue

        # Model XI
        model_xi = optimize_xi_for_gw(gw_data[["name", "team", "position", "value", "pred_pts"]], budget=100.0)
        if len(model_xi) == 0: continue
        actual_pts_model = gw_data[gw_data["name"].isin(model_xi["name"])]["total_points"].sum()

        # Baseline top ownership
        top_own = gw_data.nlargest(50, "selected").sort_values("pred_pts", ascending=False)
        try:
            baseline_own = optimize_xi_for_gw(
                top_own[["name", "team", "position", "value", "pred_pts"]].assign(pred_pts=top_own["selected"]),
                budget=100.0
            )
            actual_pts_own = gw_data[gw_data["name"].isin(baseline_own["name"])]["total_points"].sum() if len(baseline_own) > 0 else 0
        except:
            actual_pts_own = 0

        # Baseline form
        try:
            baseline_form = optimize_xi_for_gw(
                gw_data[["name", "team", "position", "value"]].assign(
                    pred_pts=gw_data["form_fpl"].fillna(0)
                ),
                budget=100.0
            )
            actual_pts_form = gw_data[gw_data["name"].isin(baseline_form["name"])]["total_points"].sum() if len(baseline_form) > 0 else 0
        except:
            actual_pts_form = 0

        # Baseline xP oficial
        try:
            baseline_xP = optimize_xi_for_gw(
                gw_data[["name", "team", "position", "value"]].assign(
                    pred_pts=gw_data["xP"].fillna(0)
                ),
                budget=100.0
            )
            actual_pts_xP = gw_data[gw_data["name"].isin(baseline_xP["name"])]["total_points"].sum() if len(baseline_xP) > 0 else 0
        except:
            actual_pts_xP = 0

        # Baseline random (media 11 jugadores random)
        np.random.seed(gw)
        random_players = gw_data.sample(min(11, len(gw_data)))
        actual_pts_random = random_players["total_points"].sum()

        results.append({
            "GW": gw,
            "Modelo LGB": actual_pts_model,
            "xP oficial FPL": actual_pts_xP,
            "Top ownership": actual_pts_own,
            "Top form": actual_pts_form,
            "Random XI": actual_pts_random,
        })
        print(f"  GW{gw}: modelo={actual_pts_model} xP={actual_pts_xP} form={actual_pts_form} own={actual_pts_own} random={actual_pts_random}")

    df_res = pd.DataFrame(results)
    print(f"\n{'═'*70}")
    print(f"  TOTALES ACUMULADOS TEMPORADA 25/26 ({len(df_res)} GWs)")
    print(f"{'═'*70}")

    totals = df_res.drop(columns=["GW"]).sum()
    for name, pts in totals.sort_values(ascending=False).items():
        pct_vs_random = (pts / totals["Random XI"] - 1) * 100 if totals["Random XI"] > 0 else 0
        marker = " ⭐" if name == "Modelo LGB" else ""
        print(f"  {name:20s} {int(pts):5d} pts  ({pct_vs_random:+.1f}% vs random){marker}")

    # Guardar
    df_res.to_csv(PROC / "backtest_2526_results.csv", index=False)
    print(f"\nDetalle GW por GW guardado: {PROC / 'backtest_2526_results.csv'}")


if __name__ == "__main__":
    main()
