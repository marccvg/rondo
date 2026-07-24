"""
Fase 2 completa: merge xG Understat + build features A+B+C+E+F + LightGBM.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from elo import compute_elo_history
from features.family_a_strength import compute_family_a
from features.family_b_form import compute_family_b
from features.family_c_context import compute_family_c
from features.family_e_tactical import compute_family_e
from features.family_f_matchup import compute_family_f
from dixon_coles import matrix_to_1x2, score_matrix
from metrics import all_metrics
from fase1_baseline import home_prior_baseline, pinnacle_probs, uniform_baseline

ROOT = Path("/home/Projects/futbol_elo_poisson")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


TEAM_ALIAS_FD_TO_UND = {
    # Premier
    "Man United": "Manchester United", "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest", "Newcastle": "Newcastle United",
    "Wolves": "Wolverhampton Wanderers", "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham", "Leeds": "Leeds", "Leicester": "Leicester",
    "Tottenham": "Tottenham",
    # LaLiga
    "Ath Madrid": "Atletico Madrid", "Ath Bilbao": "Athletic Club",
    "Betis": "Real Betis", "Vallecano": "Rayo Vallecano",
    "Alaves": "Alaves", "Cadiz": "Cadiz", "Almeria": "Almeria",
    "Malaga": "Malaga", "La Coruna": "Deportivo La Coruna",
    "Leganes": "Leganes", "Sociedad": "Real Sociedad",
    "Celta": "Celta Vigo", "Espanol": "Espanyol",
    "Huesca": "SD Huesca", "Valladolid": "Real Valladolid",
    # Bundesliga
    "Ein Frankfurt": "Eintracht Frankfurt",
    "M'gladbach": "Borussia M.Gladbach",
    "FC Koln": "FC Cologne", "Fortuna Dusseldorf": "Fortuna Duesseldorf",
    "Hertha": "Hertha Berlin", "Dortmund": "Borussia Dortmund",
    "Bayern Munich": "Bayern Munich", "Leverkusen": "Bayer Leverkusen",
    "Werder Bremen": "Werder Bremen", "RB Leipzig": "RasenBallsport Leipzig",
    "Bielefeld": "Arminia Bielefeld", "Hamburg": "Hamburger SV",
    "Hannover": "Hannover 96", "Heidenheim": "FC Heidenheim",
    "Mainz": "Mainz 05", "Stuttgart": "VfB Stuttgart",
    # Ligue 1
    "Paris SG": "Paris Saint Germain", "St Etienne": "Saint-Etienne",
    "Nimes": "Nimes", "Clermont": "Clermont Foot",
    # Serie A
    "Verona": "Hellas Verona", "Napoli": "Napoli", "Roma": "Roma",
    "Inter": "Inter", "AC Milan": "AC Milan", "Milan": "AC Milan",
    "Juventus": "Juventus", "Atalanta": "Atalanta",
    "Parma": "Parma Calcio 1913", "Spal": "SPAL 2013",
}

LEAGUE_MAP = {
    "premier": "ENG-Premier League",
    "laliga": "ESP-La Liga",
    "seriea": "ITA-Serie A",
    "bundesliga": "GER-Bundesliga",
    "ligue1": "FRA-Ligue 1",
}


def merge_xg(matches: pd.DataFrame, understat: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches["_home"] = matches["home_team"].apply(lambda t: TEAM_ALIAS_FD_TO_UND.get(t, t))
    matches["_away"] = matches["away_team"].apply(lambda t: TEAM_ALIAS_FD_TO_UND.get(t, t))
    matches["_league"] = matches["league"].map(LEAGUE_MAP)
    matches["_date"] = pd.to_datetime(matches["date"]).dt.date

    und = understat.copy()
    und["_date"] = pd.to_datetime(und["date"]).dt.date
    und["_league"] = und["_league_norm"]
    und = und[["_league", "_date", "home_team", "away_team", "home_xg", "away_xg"]].rename(
        columns={"home_team": "_home", "away_team": "_away", "home_xg": "xg_home", "away_xg": "xg_away"}
    )

    # De-duplicar understat por si hay filas repetidas de fixtures
    und = und.drop_duplicates(subset=["_league", "_date", "_home", "_away"], keep="first")
    merged = matches.merge(und, on=["_league", "_date", "_home", "_away"], how="left", validate="many_to_one")
    merged = merged.reset_index(drop=True)
    n_matched = merged["xg_home"].notna().sum()
    print(f"  xG matched: {n_matched:,}/{len(merged):,} ({n_matched/len(merged)*100:.1f}%)")
    return merged.drop(columns=[c for c in merged.columns if c.startswith("_")])


def train_lgb(X_train, y_train, X_val, y_val):
    train_ds = lgb.Dataset(X_train, y_train)
    val_ds = lgb.Dataset(X_val, y_val, reference=train_ds)
    params = {
        "objective": "poisson", "metric": "poisson",
        "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 50,
        "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
        "lambda_l2": 0.1, "verbose": -1, "seed": 42,
    }
    return lgb.train(params, train_ds, num_boost_round=2000, valid_sets=[val_ds],
                     valid_names=["val"], callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])


def probs_from_lambdas(lh, la, rho=-0.03):
    probs = np.zeros((len(lh), 3))
    for i in range(len(lh)):
        m = score_matrix(max(lh[i], 0.05), max(la[i], 0.05), rho=rho, max_goals=8)
        probs[i] = matrix_to_1x2(m)
    return probs


def main():
    print(">>> Fase 2 FULL — con xG Understat")
    matches = pd.read_parquet(PROC / "matches_all.parquet")
    understat = pd.read_parquet(ROOT / "data" / "raw" / "understat_full" / "understat_schedule_5big.parquet")
    print(f"Matches: {len(matches):,}, Understat: {len(understat):,}")

    print("\n1. Merge xG...")
    df = merge_xg(matches, understat)

    print("2. Compute ELO history...")
    df = compute_elo_history(df)

    print("3. Feature engineering (5 familias)...")
    fa = compute_family_a(df); print(f"  A: {len(fa.columns)} cols")
    fb = compute_family_b(df); print(f"  B: {len(fb.columns)} cols")
    fc = compute_family_c(df); print(f"  C: {len(fc.columns)} cols")
    ff = compute_family_f(df); print(f"  F: {len(ff.columns)} cols")
    # E solo si hay xG
    df_xg = df.dropna(subset=["xg_home", "xg_away"]).copy()
    print(f"  Partidos con xG: {len(df_xg):,}/{len(df):,}")
    fe = compute_family_e(df_xg); print(f"  E: {len(fe.columns)} cols (aplicada sobre {len(fe):,})")

    # Consolidar (E puede tener NaN donde faltó xG)
    features_no_e = pd.concat([fa, fb, fc, ff], axis=1)
    # Alinear E al índice completo
    fe_full = pd.DataFrame(index=df.index, columns=fe.columns)
    fe_full.loc[df_xg.index] = fe.values
    all_feats = pd.concat([features_no_e, fe_full], axis=1)
    print(f"Total features: {len(all_feats.columns)}")

    all_feats["target_goals_home"] = df["home_goals"].to_numpy()
    all_feats["target_goals_away"] = df["away_goals"].to_numpy()
    all_feats["target_result"] = df["result"].to_numpy()
    all_feats["meta_date"] = df["date"].to_numpy()
    all_feats["meta_season"] = df["season"].to_numpy()
    all_feats["meta_pinnacle_home"] = df["pinnacle_home"].to_numpy()
    all_feats["meta_pinnacle_draw"] = df["pinnacle_draw"].to_numpy()
    all_feats["meta_pinnacle_away"] = df["pinnacle_away"].to_numpy()

    feature_cols = [c for c in all_feats.columns if not c.startswith("target_") and not c.startswith("meta_")]
    print(f"Feature cols entrenables: {len(feature_cols)}")

    # Split temporal
    train = all_feats[all_feats["meta_season"].isin([f"20{y:02d}-{y+1:02d}" for y in range(17, 23)])].copy()
    val = all_feats[all_feats["meta_season"] == "2023-24"].copy()
    test = all_feats[all_feats["meta_season"] == "2024-25"].copy()
    print(f"Train {len(train):,} | Val {len(val):,} | Test {len(test):,}")

    Xt = train[feature_cols].astype(np.float32).to_numpy()
    Xv = val[feature_cols].astype(np.float32).to_numpy()
    Xtest = test[feature_cols].astype(np.float32).to_numpy()

    print("\n4. LightGBM training (2 modelos)...")
    m_h = train_lgb(Xt, train["target_goals_home"].to_numpy(), Xv, val["target_goals_home"].to_numpy())
    m_a = train_lgb(Xt, train["target_goals_away"].to_numpy(), Xv, val["target_goals_away"].to_numpy())
    print(f"Best iter home: {m_h.best_iteration} | away: {m_a.best_iteration}")

    test_lh = m_h.predict(Xtest, num_iteration=m_h.best_iteration)
    test_la = m_a.predict(Xtest, num_iteration=m_a.best_iteration)
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
        all_metrics(test_true, test_probs, "LGB Fase2 FULL (A+B+C+E+F)"),
        all_metrics(test_true, pinnacle_probs(test_pin), "Pinnacle"),
    ]
    print("\n=== TEST 2024-25 ===")
    for m in metrics:
        print(f"  {m['model']:30s}  log-loss={m['log_loss']:.4f}  RPS={m['rps']:.4f}  Brier={m['brier']:.4f}")

    # Feature importance
    imp = pd.DataFrame({
        "feature": feature_cols,
        "imp_home": m_h.feature_importance("gain"),
        "imp_away": m_a.feature_importance("gain"),
    })
    imp["imp_total"] = imp["imp_home"] + imp["imp_away"]
    print("\n=== TOP 15 features ===")
    print(imp.sort_values("imp_total", ascending=False).head(15).to_string(index=False))

    # Reporte
    report = REPORTS / "fase2_full.md"
    with open(report, "w") as f:
        f.write("# Fase 2 FULL — LightGBM con familia E (xG) integrada\n\n")
        f.write(f"Generado: {pd.Timestamp.now().isoformat()}\n\n")
        f.write(f"## Setup\n- Features: {len(feature_cols)}\n- Train {len(train):,}, Val {len(val):,}, Test {len(test):,}\n- Best iter home: {m_h.best_iteration}, away: {m_a.best_iteration}\n\n")
        f.write("## Métricas TEST 2024-25\n\n| Modelo | log-loss | RPS | Brier |\n|---|---|---|---|\n")
        for m in metrics:
            f.write(f"| {m['model']} | {m['log_loss']:.4f} | {m['rps']:.4f} | {m['brier']:.4f} |\n")
        model_ll = next(m['log_loss'] for m in metrics if 'FULL' in m['model'])
        pin_ll = next(m['log_loss'] for m in metrics if m['model'] == 'Pinnacle')
        f.write(f"\n## Comparación vs Fase 1 (0.9859) y Fase 2 básica (0.9888)\n\n")
        f.write(f"- Fase 2 FULL log-loss test: **{model_ll:.4f}**\n")
        f.write(f"- Mejora vs Fase 1 baseline: **{(0.9859-model_ll)/0.9859*100:+.2f}%**\n")
        f.write(f"- Mejora vs Fase 2 básica (sin xG): **{(0.9888-model_ll)/0.9888*100:+.2f}%**\n")
        f.write(f"- Ratio vs Pinnacle: **{model_ll/pin_ll:.4f}**\n\n")
        f.write("## Top 15 features\n\n")
        f.write(imp.sort_values("imp_total", ascending=False).head(15).to_markdown(index=False))
    print(f"\nReporte: {report}")


if __name__ == "__main__":
    main()
