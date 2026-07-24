# futbol_elo_poisson

**A football result prediction system combining a recursive ELO rating, a Poisson bivariate goal model with the Dixon-Coles low-score correction, and LightGBM gradient boosting on ~60-120 engineered features.**

Data: 5 top European leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1), 2017/18 - 2024/25.
Sources: 100% public and free (Football-Data.co.uk, FBref via `soccerdata`, ClubELO, StatsBomb Open Data).

## Baseline result (Phase 1, 2026-07-24)

Out-of-sample test on the full 2024-25 season (1,752 matches across 5 leagues):

| Model | Log-loss | RPS | Brier |
|---|---|---|---|
| Uniform (1/3, 1/3, 1/3) | 1.0986 | 0.4724 | 0.6667 |
| Home-prior (empirical home-win rate) | 1.0775 | 0.4655 | 0.6527 |
| **ELO + Poisson + Dixon-Coles** | **0.9859** | **0.4015** | **0.5880** |
| Pinnacle closing odds (practical ceiling) | 0.9606 | 0.3870 | 0.5706 |

**Ratio model/Pinnacle = 1.026** — the baseline is within 2.6% of Pinnacle's log-loss using **a single feature** (ELO difference). The upside for Phase 2 feature engineering (xG rolling, form, squad state, matchup) is closing that 2.6% gap.

## Project status

- ✅ **Phase 0** — Ingest (Football-Data.co.uk 40 CSVs, ClubELO, StatsBomb Open sparse, FBref probe)
- ✅ **Phase 1** — Baseline: recursive ELO + Poisson GLM + Dixon-Coles + rigorous temporal split + comparison against Pinnacle
- 🚧 **Phase 2** — Feature engineering (7 families: strength, form, context, squad, tactical, matchup, external)
- ⏳ **Phase 3** — LightGBM (Poisson objective) + Optuna + SHAP
- ⏳ **Phase 4** — Full bivariate distribution + market derivation (1X2, O/U, BTTS, exact score, handicap) + calibration
- ⏳ **Phase 5** — Rigorous backtest 3 seasons vs Pinnacle + Monte Carlo season simulation
- ⏳ **Phase 6** — Streamlit dashboard + auto-refresh

Detailed plan and progress log: [PLAN.md](PLAN.md).

## What this project is

- A **reproducible, honest baseline** for football match prediction using only public data.
- An **educational reference** showing the correct pipeline: temporal split, ELO recursion, Poisson GLM with log-link, Dixon-Coles correction, proper scoring rules for probabilistic classification.
- A **portfolio piece** demonstrating end-to-end applied ML: data ingest, feature engineering, model training, calibration, backtest, deployment.

## What this project is not

- Not a betting system. Pinnacle's closing odds beat this baseline systematically; add bookmaker margin and you lose. This is educational.
- Not competitive with Wyscout / StatsBomb IQ commercial platforms — those have proprietary tracking data. This uses public aggregated data.
- Not using deep learning. Dataset is ~14K matches — gradient boosting outperforms neural networks by a wide margin here and is far more interpretable.

## Stack

Python 3.11+ · pandas · duckdb · soccerdata · statsmodels · scipy · scikit-learn · LightGBM · Optuna · SHAP · Streamlit · pytest.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Phase 0 — ingest
python src/ingest_fase0.py

# Phase 1 — baseline
python src/fase1_baseline.py
# → reports/fase1_baseline.md with metrics
```

## Repository structure

```
futbol_elo_poisson/
├── README.md
├── PLAN.md                 # phased plan with atomic tasks
├── CHANGELOG.md
├── LICENSE                 # MIT
├── requirements.txt
├── data/
│   ├── raw/                # source data (gitignored)
│   ├── processed/          # consolidated parquets (gitignored)
│   └── INVENTORY.md        # what's in raw/, coverage per source
├── notebooks/              # exploratory notebooks (Phase 2+)
├── src/
│   ├── data_loader.py      # Football-Data.co.uk normalization
│   ├── elo.py              # recursive ELO with goal-diff multiplier
│   ├── dixon_coles.py      # bivariate Poisson + DC low-score correction
│   ├── fase1_baseline.py   # end-to-end Phase 1 pipeline
│   ├── ingest_fase0.py     # Phase 0 ingestion
│   └── ingest_fbref_full.py # FBref advanced stats (Phase 2 prep)
├── models/                 # LightGBM pickles (Phase 3+)
├── reports/                # backtest reports
└── tests/                  # pytest unit tests
```

## Methodology

### 1. ELO recursive rating
Every team starts at 1500. After each match: `Δ = K × goal_diff_multiplier × (actual − expected)` with K=20, home advantage = 65 Elo points. Goal-difference multiplier: 1 goal → ×1.0, 2 → ×1.5, ≥3 → ×(11+gd)/8. Run recursively on 14,359 matches sorted chronologically.

### 2. Temporal split (NEVER random)
- **Train**: 2017/18 - 2022/23 (10,855 matches)
- **Val**: 2023/24 (1,752)
- **Test**: 2024/25 (1,752, never seen)

Random CV would leak future information and yield falsely optimistic numbers.

### 3. Poisson GLM
Two separate GLMs (statsmodels), Poisson family, log link:
- `log(λ_home) = α_H + β_H · elo_diff`
- `log(λ_away) = α_A + β_A · elo_diff`

Coefficients are learned by maximum likelihood — no manual weighting.

### 4. Dixon-Coles correction
Independent Poisson under-models low scores. Dixon-Coles (1997) corrects P(0,0), P(1,0), P(0,1), P(1,1) with a `τ(i,j,λH,λA,ρ)` factor. Optimal ρ found by grid search on training samples: **ρ = -0.030** (consistent with the original 1997 paper's range of -0.02 to -0.08).

### 5. Probabilistic metrics
- **Log-loss** — standard proper scoring rule for probabilistic classification.
- **RPS (Ranked Probability Score)** — proper scoring rule that penalizes misordered predictions across the natural H-D-A order.
- **Brier score** — squared error of the probability vector against one-hot ground truth.

## References

- Dixon, M.J. & Coles, S.G. (1997). *Modelling association football scores and inefficiencies in the football betting market*. Journal of the Royal Statistical Society, Series C.
- Baio, G. & Blangiardo, M. (2010). *Bayesian hierarchical model for the prediction of football results*. Journal of Applied Statistics.
- Karlis, D. & Ntzoufras, I. (2003). *Analysis of sports data by using bivariate Poisson models*. Journal of the Royal Statistical Society, Series D.
- soccerdata: https://github.com/probberechts/soccerdata
- FBref: https://fbref.com (data provided by StatsBomb from 2017/18)
- Pinnacle closing odds via Football-Data.co.uk: https://www.football-data.co.uk

## Roadmap

- **Phase 2**: 100+ features across strength, form, context, squad state, tactical style, matchup, external.
- **Phase 3**: LightGBM with Poisson objective, Optuna hyperparameter search, SHAP interpretability.
- **Phase 4**: Full match probability distribution derived from λ_home, λ_away → derive markets (1X2, O/U, BTTS, exact score, handicap).
- **Phase 5**: 3-season backtest against Pinnacle, Monte Carlo season simulation (title, top-4, relegation probabilities), KO bracket simulation.
- **Phase 6**: Streamlit dashboard with next-round predictions auto-refresh, historical tracking, per-match SHAP explainability.

## Future extensions

- **Champions League**: not in soccerdata's built-in `league_dict`. Can be added via `~/soccerdata/config/league_dict.json` custom entry pointing to `https://fbref.com/en/comps/8/Champions-League-Stats`. Deferred to post-Phase 5.
- **Additional leagues**: Championship, Segunda División, Serie B, Ligue 2, 2. Bundesliga — extend by adding to `LEAGUES` in `src/ingest_fbref_full.py`.
- **Event-level features**: expected threat (xT), player embeddings, chemistry models — require StatsBomb events (~1.5 GB, not currently downloaded); Phase 3+ candidates.

## License

MIT — see [LICENSE](LICENSE).

## Contributing

This is a personal educational project. Issues and PRs are welcome. Please open an issue first for large changes.
