"""
Fase 5 — Backtest riguroso 3 temporadas + simulación Monte Carlo por liga.

Cada temporada test entrenamos con TODAS las temporadas previas.
- 2022/23: train 17/18-21/22 (5 temp)
- 2023/24: train 17/18-22/23 (6 temp)
- 2024/25: train 17/18-23/24 (7 temp)

Genera:
- Métricas por temporada test × liga
- Simulación Monte Carlo de LaLiga 24/25 → probabilidad campeón/top4/descenso
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dixon_coles import matrix_to_1x2, score_matrix
from fase1_baseline import pinnacle_probs
from fase3_optuna_shap import build_dataset
from metrics import all_metrics

ROOT = Path("/home/Projects/futbol_elo_poisson")
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

TEST_SEASONS = ["2022-23", "2023-24", "2024-25"]

LGB_PARAMS = {
    "objective": "poisson",
    "metric": "poisson",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": 42,
}


def train_on(X_train, y_train, X_val, y_val):
    train_ds = lgb.Dataset(X_train, y_train)
    val_ds = lgb.Dataset(X_val, y_val, reference=train_ds)
    return lgb.train(
        LGB_PARAMS, train_ds, num_boost_round=2000,
        valid_sets=[val_ds], valid_names=["val"],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )


def probs_from_lambdas(lh, la, rho=-0.03):
    probs = np.zeros((len(lh), 3))
    for i in range(len(lh)):
        m = score_matrix(max(lh[i], 0.05), max(la[i], 0.05), rho=rho, max_goals=8)
        probs[i] = matrix_to_1x2(m)
    return probs


def backtest_one_season(df, feature_cols, test_season):
    """Entrena con todo pre-test y evalúa en test_season."""
    all_seasons = sorted(df["meta_season"].unique())
    train_seasons = [s for s in all_seasons if s < test_season]
    if len(train_seasons) < 2:
        return None
    train = df[df["meta_season"].isin(train_seasons[:-1])].copy()
    val = df[df["meta_season"] == train_seasons[-1]].copy()  # última temp previa como val
    test = df[df["meta_season"] == test_season].copy()

    Xt = train[feature_cols].astype(np.float32).to_numpy()
    Xv = val[feature_cols].astype(np.float32).to_numpy()
    Xtest = test[feature_cols].astype(np.float32).to_numpy()

    m_h = train_on(Xt, train["target_goals_home"].to_numpy(), Xv, val["target_goals_home"].to_numpy())
    m_a = train_on(Xt, train["target_goals_away"].to_numpy(), Xv, val["target_goals_away"].to_numpy())

    lh = m_h.predict(Xtest)
    la = m_a.predict(Xtest)
    probs = probs_from_lambdas(lh, la)

    y_true = test["target_result"].to_numpy()
    pin = test.rename(columns={
        "meta_pinnacle_home": "pinnacle_home",
        "meta_pinnacle_draw": "pinnacle_draw",
        "meta_pinnacle_away": "pinnacle_away",
    })
    m_model = all_metrics(y_true, probs, f"Modelo test {test_season}")
    m_pin = all_metrics(y_true, pinnacle_probs(pin), f"Pinnacle test {test_season}")

    return {
        "season": test_season,
        "n": len(test),
        "model_ll": m_model["log_loss"],
        "model_rps": m_model["rps"],
        "pinnacle_ll": m_pin["log_loss"],
        "pinnacle_rps": m_pin["rps"],
        "ratio": m_model["log_loss"] / m_pin["log_loss"],
        "test_probs": probs,
        "test": test,
        "test_lh": lh,
        "test_la": la,
    }


def montecarlo_league(test, lh, la, league_name, n_sims=5000):
    """Simulación Monte Carlo temporada completa para una liga concreta."""
    # test es DataFrame; necesitamos filtrar por liga (usando pinnacle o meta cols)
    # meta cols disponibles: meta_date, meta_season, pinnacle_*. No hay meta_league.
    # Voy a re-obtener liga vía join con matches_all
    matches_all = pd.read_parquet(ROOT / "data/processed/matches_all.parquet")
    matches_all["_key"] = matches_all["date"].astype(str) + "|" + matches_all["home_team"] + "|" + matches_all["away_team"]
    # Necesito home_team, away_team en test; no están en test (son solo feature cols + target)
    # Reconstruyo via merge_xg... más simple: skip filtro por liga, hacer sobre todos los partidos test
    pass


def monte_carlo_full_season_by_league(test, lh, la, matches_all_source, league_key, n_sims=5000):
    """Filtra test por liga y simula temporada."""
    # Reconstruir home_team / away_team desde matches_all
    m = matches_all_source.copy()
    m["_date"] = pd.to_datetime(m["date"])
    test = test.copy()
    test["_date"] = pd.to_datetime(test["meta_date"])
    test["idx_orig"] = np.arange(len(test))
    # join por date + pinnacle_home unique
    merged = test.merge(
        m[["_date", "league", "home_team", "away_team", "home_goals", "away_goals", "pinnacle_home"]],
        left_on=["_date", "meta_pinnacle_home"],
        right_on=["_date", "pinnacle_home"],
        how="left",
    )
    merged = merged.drop_duplicates(subset=["idx_orig"], keep="first").sort_values("idx_orig").reset_index(drop=True)
    liga_mask = merged["league"] == league_key
    if liga_mask.sum() == 0:
        return None

    lh_liga = lh[liga_mask.to_numpy()]
    la_liga = la[liga_mask.to_numpy()]
    partidos_liga = merged[liga_mask].reset_index(drop=True)

    teams = sorted(set(partidos_liga["home_team"]) | set(partidos_liga["away_team"]))
    n_partidos = len(partidos_liga)
    print(f"    {league_key}: {n_partidos} partidos, {len(teams)} equipos")

    rng = np.random.default_rng(42)
    champion_c = Counter()
    top4_c = Counter()
    releg_c = Counter()

    for _ in range(n_sims):
        pts = defaultdict(int)
        for i in range(n_partidos):
            gh = rng.poisson(lh_liga[i])
            ga = rng.poisson(la_liga[i])
            h = partidos_liga.at[i, "home_team"]
            a = partidos_liga.at[i, "away_team"]
            if gh > ga: pts[h] += 3
            elif gh < ga: pts[a] += 3
            else:
                pts[h] += 1; pts[a] += 1
        ranked = sorted(pts.items(), key=lambda kv: -kv[1])
        champion_c[ranked[0][0]] += 1
        for t, _ in ranked[:4]:
            top4_c[t] += 1
        for t, _ in ranked[-3:]:
            releg_c[t] += 1

    # Ranking real
    pts_real = defaultdict(int)
    for _, row in partidos_liga.iterrows():
        if row["home_goals"] > row["away_goals"]: pts_real[row["home_team"]] += 3
        elif row["home_goals"] < row["away_goals"]: pts_real[row["away_team"]] += 3
        else:
            pts_real[row["home_team"]] += 1; pts_real[row["away_team"]] += 1
    ranked_real = sorted(pts_real.items(), key=lambda kv: -kv[1])

    return {
        "champion": champion_c,
        "top4": top4_c,
        "relegation": releg_c,
        "real_ranking": ranked_real,
        "n_sims": n_sims,
    }


def main():
    print(">>> Fase 5 — Backtest 3 temporadas + Monte Carlo LaLiga 24/25")
    df = build_dataset()
    feature_cols = [c for c in df.columns if not c.startswith("target_") and not c.startswith("meta_")]

    print(f"\n=== Backtest por temporada ===")
    results = []
    for season in TEST_SEASONS:
        print(f"\nProcesando {season}...")
        r = backtest_one_season(df, feature_cols, season)
        if r:
            results.append(r)
            print(f"  Modelo log-loss: {r['model_ll']:.4f}  |  Pinnacle: {r['pinnacle_ll']:.4f}  |  ratio: {r['ratio']:.4f}")

    # Tabla resumen
    print("\n=== Resumen backtest ===")
    print(f"{'Season':<10} {'N':>6} {'Model_LL':>10} {'Pin_LL':>10} {'Ratio':>8} {'Model_RPS':>10} {'Pin_RPS':>10}")
    for r in results:
        print(f"{r['season']:<10} {r['n']:>6} {r['model_ll']:>10.4f} {r['pinnacle_ll']:>10.4f} {r['ratio']:>8.4f} {r['model_rps']:>10.4f} {r['pinnacle_rps']:>10.4f}")

    avg_ratio = np.mean([r["ratio"] for r in results])
    print(f"\nRatio promedio vs Pinnacle: {avg_ratio:.4f}")

    # Monte Carlo LaLiga 24/25 (última temporada test)
    last_r = results[-1]
    print("\n=== Monte Carlo LaLiga 2024/25 ===")
    matches_all = pd.read_parquet(ROOT / "data/processed/matches_all.parquet")
    mc = monte_carlo_full_season_by_league(
        last_r["test"], last_r["test_lh"], last_r["test_la"],
        matches_all, "laliga", n_sims=5000,
    )
    if mc:
        print(f"\n🏆 CAMPEÓN PROB (top 10):")
        for team, w in mc["champion"].most_common(10):
            print(f"  {team:20s} {w/mc['n_sims']*100:5.1f}%")

        print(f"\n🥇 TOP-4 PROB (Champions):")
        for team, c in mc["top4"].most_common(10):
            print(f"  {team:20s} {c/mc['n_sims']*100:5.1f}%")

        print(f"\n📉 DESCENSO PROB:")
        for team, c in mc["relegation"].most_common(10):
            print(f"  {team:20s} {c/mc['n_sims']*100:5.1f}%")

        print(f"\n─── Ranking REAL LaLiga 24/25 ───")
        for i, (t, p) in enumerate(mc["real_ranking"][:10], 1):
            camp_prob = mc["champion"][t] / mc["n_sims"] * 100
            print(f"  {i:2d}. {t:20s} {p:3d} pts (modelo predijo campeón con {camp_prob:.1f}%)")

    # Reporte
    report = REPORTS / "fase5_backtest.md"
    with open(report, "w") as f:
        f.write("# Fase 5 — Backtest 3 temporadas + Monte Carlo\n\n")
        f.write(f"Generado: {pd.Timestamp.now().isoformat()}\n\n")
        f.write("## Setup\n\n- Cada temporada test: entrenamos con todas las previas + última previa como val (early stopping).\n\n")
        f.write("## Resumen por temporada\n\n")
        f.write("| Season | N partidos | Model LL | Pinnacle LL | Ratio |\n|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['season']} | {r['n']} | {r['model_ll']:.4f} | {r['pinnacle_ll']:.4f} | {r['ratio']:.4f} |\n")
        f.write(f"\n**Ratio promedio: {avg_ratio:.4f}** (1.000 = igual a Pinnacle).\n\n")

        if mc:
            f.write("## Monte Carlo LaLiga 24/25 (5.000 simulaciones)\n\n")
            f.write("### Prob campeón (top 10)\n\n")
            f.write("| Equipo | Prob |\n|---|---|\n")
            for team, w in mc["champion"].most_common(10):
                f.write(f"| {team} | {w/mc['n_sims']*100:.1f}% |\n")
            f.write("\n### Prob top-4\n\n| Equipo | Prob |\n|---|---|\n")
            for team, c in mc["top4"].most_common(10):
                f.write(f"| {team} | {c/mc['n_sims']*100:.1f}% |\n")
            f.write("\n### Prob descenso\n\n| Equipo | Prob |\n|---|---|\n")
            for team, c in mc["relegation"].most_common(10):
                f.write(f"| {team} | {c/mc['n_sims']*100:.1f}% |\n")
            f.write("\n### Ranking real LaLiga 24/25\n\n| Pos | Equipo | Pts | Prob modelo=campeón |\n|---|---|---|---|\n")
            for i, (t, p) in enumerate(mc["real_ranking"][:10], 1):
                cp = mc["champion"][t] / mc["n_sims"] * 100
                f.write(f"| {i} | {t} | {p} | {cp:.1f}% |\n")
    print(f"\nReporte: {report}")


if __name__ == "__main__":
    main()
