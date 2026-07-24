# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### In progress
- Phase 2 — feature engineering (7 families): FBref advanced stats ingestion running in background.

## [0.1.0] — 2026-07-24

Initial baseline. Phases 0 and 1 completed same day.

### Added
- Phase 0 ingestion (`src/ingest_fase0.py`):
  - Football-Data.co.uk: 40 CSVs, 5 leagues × 8 seasons (14,359 matches, 100% Pinnacle closing odds coverage).
  - ClubELO snapshot.
  - StatsBomb Open Data via sparse-checkout: `competitions.json` + 80 match JSONs + 4,235 lineup JSONs (104 MB total; events deferred until needed).
  - FBref and Understat probes validated soccerdata scraping works in this environment.
- Data loader (`src/data_loader.py`): normalizes 40 heterogeneous CSVs into a single Parquet with unified schema (date, league, season, teams, goals, result, Pinnacle odds handling historical column name changes).
- Recursive ELO module (`src/elo.py`): classical ELO with home advantage (+65 points) and goal-difference multiplier. Sanity-check ranking coherent with reality (PSG, Inter, Bayern, Barça, Madrid, Liverpool, City, Arsenal, Leverkusen top 10).
- Dixon-Coles bivariate Poisson (`src/dixon_coles.py`): 9×9 score matrix with low-score correction on cells (0,0), (1,0), (0,1), (1,1). Market derivation utilities: 1X2, over/under, BTTS.
- Phase 1 end-to-end pipeline (`src/fase1_baseline.py`): temporal split → Poisson GLM → optimized ρ Dixon-Coles → probabilistic metrics vs baselines and Pinnacle → auto-generated report.

### Results (Phase 1 baseline, test on 2024-25, 1,752 matches)
| Model | Log-loss | RPS | Brier |
|---|---|---|---|
| Uniform | 1.0986 | 0.4724 | 0.6667 |
| Home prior | 1.0775 | 0.4655 | 0.6527 |
| **ELO+Poisson+DC (this)** | **0.9859** | **0.4015** | **0.5880** |
| Pinnacle | 0.9606 | 0.3870 | 0.5706 |

Model within **2.6%** of Pinnacle's log-loss using a **single feature** (ELO diff).

### Design decisions
- Temporal split (train 17/18-22/23, val 23/24, test 24/25) — never random.
- Two separate GLMs (goals_home, goals_away) — simpler than a single conditional model, easier to interpret.
- Grid-search optimization of ρ Dixon-Coles on a training sub-sample (n=2000) for speed — final ρ = -0.030, consistent with Dixon-Coles (1997) paper's range.
- 5 top European leagues only in v1 (Champions League not available in soccerdata's built-in `league_dict` — deferred to a future extension).
