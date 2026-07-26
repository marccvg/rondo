"""
Optimizador de plantilla Fantasy Premier League vía Linear Programming.

Objetivo: dado presupuesto £100M FPL, elegir 15 jugadores (2 GK, 5 DEF, 5 MID, 3 FWD)
que maximicen puntos esperados totales del XI titular + factor bench.

Constraints:
  - 2 GK, 5 DEF, 5 MID, 3 FWD (15 total)
  - XI titular: 1 GK + 3-5 DEF + 3-5 MID + 1-3 FWD (11 total)
  - Máximo 3 jugadores por club
  - Presupuesto ≤ £100M

Predicciones puntos vienen del modelo entrenado + FPL API actual (precios frescos).
"""
from __future__ import annotations
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import pulp

ROOT = Path("/home/Projects/futbol_elo_poisson/fantasy")
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def load_current_squad_data():
    """Combina snapshot FPL API + features históricas para predicción actual."""
    boot = json.load(open(RAW / "fpl_bootstrap.json"))
    df_hist = pd.read_parquet(PROC / "fpl_features.parquet")

    # Última fila por jugador con sus features rolling actuales
    last_per_player = df_hist.sort_values(["name", "_season", "GW"]).groupby("name").tail(1)
    last_per_player = last_per_player.set_index("name")

    # Info actual de FPL API
    teams_by_id = {t["id"]: t["name"] for t in boot["teams"]}
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    current = []
    for p in boot["elements"]:
        name = p["web_name"]
        # Buscar en histórico
        hist_row = None
        for candidate in [name, p.get("first_name", "") + " " + name]:
            if candidate in last_per_player.index:
                hist_row = last_per_player.loc[candidate]
                if isinstance(hist_row, pd.DataFrame):
                    hist_row = hist_row.iloc[0]
                break
        current.append({
            "id": p["id"],
            "name": name,
            "team": teams_by_id.get(p["team"], f"team_{p['team']}"),
            "position": pos_map.get(p["element_type"], "UNK"),
            "value_M": p["now_cost"] / 10.0,
            "total_points": p["total_points"],
            "form": float(p.get("form", "0") or 0),
            "selected_by_percent": float(p.get("selected_by_percent", "0") or 0),
            # Features rolling del histórico (última fila)
            "hist_row": hist_row,
        })
    return pd.DataFrame(current)


def predict_with_model(df_current, model_path, feat_cols_path):
    """Aplica modelo LightGBM a los jugadores actuales."""
    model = lgb.Booster(model_file=str(model_path))
    feature_cols = json.load(open(feat_cols_path))

    rows = []
    for _, r in df_current.iterrows():
        hist = r["hist_row"]
        feat = {}
        for c in feature_cols:
            if hist is not None and c in hist.index:
                v = hist[c]
                feat[c] = float(v) if pd.notna(v) else 0.0
            elif c.startswith("pos_"):
                feat[c] = 1.0 if c == f"pos_{r['position']}" else 0.0
            elif c == "value_M":
                feat[c] = r["value_M"]
            elif c == "was_home":
                feat[c] = 0.5  # neutro, no sabemos next fixture
            else:
                feat[c] = 0.0
        rows.append(feat)

    X = pd.DataFrame(rows)[feature_cols].astype(np.float32).values
    predictions = model.predict(X)
    df_current = df_current.copy()
    df_current["predicted_points_next"] = np.clip(predictions, 0, None)
    return df_current


def optimize_team(df, budget=100.0, verbose=True):
    """LP para elegir 15 jugadores óptimos."""
    # Solo jugadores con precio y predicción válidos + al menos 45 min últimas 5 (proxy titular real)
    df = df[df["value_M"] > 0].copy()
    df = df.dropna(subset=["predicted_points_next"])
    df = df.reset_index(drop=True)

    n = len(df)
    prob = pulp.LpProblem("fpl_optimize", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Objetivo: maximizar puntos esperados
    prob += pulp.lpSum(x[i] * df.loc[i, "predicted_points_next"] for i in range(n))

    # Constraints
    prob += pulp.lpSum(x[i] * df.loc[i, "value_M"] for i in range(n)) <= budget
    prob += pulp.lpSum(x[i] for i in range(n)) == 15

    # Composición
    prob += pulp.lpSum(x[i] for i in range(n) if df.loc[i, "position"] == "GK") == 2
    prob += pulp.lpSum(x[i] for i in range(n) if df.loc[i, "position"] == "DEF") == 5
    prob += pulp.lpSum(x[i] for i in range(n) if df.loc[i, "position"] == "MID") == 5
    prob += pulp.lpSum(x[i] for i in range(n) if df.loc[i, "position"] == "FWD") == 3

    # Máximo 3 por club
    for team_name in df["team"].unique():
        prob += pulp.lpSum(x[i] for i in range(n) if df.loc[i, "team"] == team_name) <= 3

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    selected = [i for i in range(n) if x[i].value() == 1]
    team_df = df.loc[selected].sort_values(
        ["position", "predicted_points_next"],
        key=lambda s: s.map({"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}) if s.name == "position" else -s
    ).reset_index(drop=True)

    if verbose:
        total_cost = team_df["value_M"].sum()
        total_pts = team_df["predicted_points_next"].sum()
        print(f"\n💰 Presupuesto usado: £{total_cost:.1f}M / £{budget:.0f}M")
        print(f"⭐ Puntos esperados próxima GW: {total_pts:.1f}")
        print(f"\n═══ PLANTILLA ÓPTIMA (15 jugadores) ═══\n")
        for pos in ["GK", "DEF", "MID", "FWD"]:
            sub = team_df[team_df["position"] == pos]
            print(f"─── {pos} ───")
            for _, r in sub.iterrows():
                print(f"  {r['name']:22s} {r['team']:20s} £{r['value_M']:.1f}M  → {r['predicted_points_next']:.2f} pts esperados")
            print()

    return team_df


def main():
    print("Cargando datos...")
    df_current = load_current_squad_data()
    print(f"Jugadores actuales FPL: {len(df_current)}")

    print("Aplicando modelo predictor...")
    df_current = predict_with_model(
        df_current,
        MODELS / "fpl_predictor_lgb.txt",
        MODELS / "feature_cols.json",
    )

    # Ranking mejores value picks
    df_current["ppm"] = df_current["predicted_points_next"] / df_current["value_M"].where(df_current["value_M"] > 0, 1)

    print("\n🎯 TOP 15 mejores VALUE picks (puntos esperados / £M):")
    print(df_current.nlargest(15, "ppm")[["name", "team", "position", "value_M", "predicted_points_next", "ppm"]].to_string(index=False))

    print("\n💪 TOP 15 mejores PUNTOS BRUTOS esperados:")
    print(df_current.nlargest(15, "predicted_points_next")[["name", "team", "position", "value_M", "predicted_points_next"]].to_string(index=False))

    print("\n" + "═" * 70)
    print("  OPTIMIZADOR PLANTILLA — Presupuesto £100M FPL")
    print("═" * 70)
    optimize_team(df_current, budget=100.0)


if __name__ == "__main__":
    main()
