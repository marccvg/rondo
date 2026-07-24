"""
MINI paso 4 — LightGBM Poisson.

Solo con features derivadas de los 4 datos (equipos + goles):
- elo_diff
- forma_home_ultimos5 (goles marcados)
- forma_away_ultimos5 (goles marcados)
- forma_home_conceded5
- forma_away_conceded5

Muestra los primeros splits del árbol para ver cómo aprende.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb

df = pd.read_parquet("/home/Projects/futbol_elo_poisson/mini/mini_matches_with_elo.parquet")
df = df.sort_values("date").reset_index(drop=True)

# Feature engineering minimal
def rolling_stat(df, team_col, val_col, window, new_col):
    df = df.sort_values("date").reset_index(drop=True)
    # Long format
    long = pd.concat([
        df[["date", "home_team", "home_goals", "away_goals"]].rename(
            columns={"home_team": "team", "home_goals": "gs", "away_goals": "gc"}
        ),
        df[["date", "away_team", "away_goals", "home_goals"]].rename(
            columns={"away_team": "team", "away_goals": "gs", "home_goals": "gc"}
        )
    ]).sort_values(["team", "date"]).reset_index(drop=True)
    long["gs_r5"] = long.groupby("team")["gs"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    long["gc_r5"] = long.groupby("team")["gc"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    return long.set_index(["team", "date"])

long_idx = rolling_stat(df, None, None, 5, None)
h = long_idx.reindex(list(zip(df["home_team"], df["date"])))
a = long_idx.reindex(list(zip(df["away_team"], df["date"])))
df["forma_home_gs5"] = h["gs_r5"].to_numpy()
df["forma_home_gc5"] = h["gc_r5"].to_numpy()
df["forma_away_gs5"] = a["gs_r5"].to_numpy()
df["forma_away_gc5"] = a["gc_r5"].to_numpy()

features = ["elo_diff", "forma_home_gs5", "forma_home_gc5", "forma_away_gs5", "forma_away_gc5"]

print("=" * 60)
print("  MINI paso 4 — LIGHTGBM POISSON")
print("=" * 60)
print(f"\nFeatures usadas ({len(features)}): {features}")

# Split
train = df[df["date"] < "2023-07-01"].copy()
val = df[(df["date"] >= "2023-07-01") & (df["date"] < "2024-07-01")].copy()
test = df[df["date"] >= "2024-07-01"].copy()
print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

# LightGBM entrenamiento con parámetros didácticos (árboles pequeños)
params = {
    "objective": "poisson",
    "metric": "poisson",
    "learning_rate": 0.05,
    "num_leaves": 15,          # árboles pequeños para poder inspeccionar
    "max_depth": 4,
    "min_data_in_leaf": 50,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "verbose": -1,
    "seed": 42,
}

print("\n─── Entrenando modelo HOME ───")
train_ds = lgb.Dataset(train[features], train["home_goals"])
val_ds = lgb.Dataset(val[features], val["home_goals"], reference=train_ds)
m_h = lgb.train(params, train_ds, num_boost_round=500,
                valid_sets=[val_ds], valid_names=["val"],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
print(f"Best iteration: {m_h.best_iteration}  (paró early stopping)")
print(f"Val loss (poisson): {m_h.best_score['val']['poisson']:.5f}")

print("\n─── Estructura del ÁRBOL 1 (el primero de todos) ───")
tree0 = m_h.dump_model()["tree_info"][0]["tree_structure"]
def print_tree(node, depth=0):
    prefix = "   " * depth
    if "leaf_value" in node:
        # Es hoja
        print(f"{prefix}HOJA valor={node['leaf_value']:+.5f}  (contribuye a log(λ))")
    else:
        feat_idx = node["split_feature"]
        feat_name = features[feat_idx]
        thr = node["threshold"]
        gain = node["split_gain"]
        print(f"{prefix}SPLIT: {feat_name} ≤ {thr:.3f} ? (gain={gain:.2f})")
        print(f"{prefix} ├─ Sí (izq):")
        print_tree(node["left_child"], depth + 2)
        print(f"{prefix} └─ No (der):")
        print_tree(node["right_child"], depth + 2)
print_tree(tree0)

print("\n─── Importance (gain) sumando TODOS los árboles ───")
imp = pd.DataFrame({
    "feature": features,
    "importance_gain": m_h.feature_importance("gain"),
    "importance_split": m_h.feature_importance("split"),
})
print(imp.sort_values("importance_gain", ascending=False).to_string(index=False))

print("\n─── Predicción concreta ───")
sample = test.iloc[[0]]
lam_h = m_h.predict(sample[features])[0]
print(f"Partido test #0: {sample['home_team'].iloc[0]} vs {sample['away_team'].iloc[0]}")
print(f"  Features: {sample[features].to_dict(orient='records')[0]}")
print(f"  λ_home predicho: {lam_h:.3f} goles esperados home")
print(f"  Goles reales: {sample['home_goals'].iloc[0]}")

# Guardar model y df
m_h.save_model("/home/Projects/futbol_elo_poisson/mini/mini_lgb_home.txt")
print("\nModelo home guardado.")

# También away
print("\n─── Entrenando modelo AWAY (silencioso, solo métrica) ───")
train_ds_a = lgb.Dataset(train[features], train["away_goals"])
val_ds_a = lgb.Dataset(val[features], val["away_goals"], reference=train_ds_a)
m_a = lgb.train(params, train_ds_a, num_boost_round=500,
                valid_sets=[val_ds_a], valid_names=["val"],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
print(f"Best iter away: {m_a.best_iteration}  |  Val loss: {m_a.best_score['val']['poisson']:.5f}")
m_a.save_model("/home/Projects/futbol_elo_poisson/mini/mini_lgb_away.txt")
df.to_parquet("/home/Projects/futbol_elo_poisson/mini/mini_matches_with_features.parquet")
print("Modelo away + df con features guardados.")
