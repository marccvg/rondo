# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] — 2026-07-24

### v1 COMPLETADO — pipeline end-to-end funcional

**Fase 3 — Optuna + SHAP**: 30 trials × 2 modelos. Optuna NO mejoró (0.9831 vs 0.9818 Fase 2 FULL) — plateau natural con datos disponibles, confirmado. SHAP top 5: `strength_elo_diff` (0.121), `strength_home_gs_r20` (0.051), `strength_elo_prob_home` (0.047), `tactical_xg_scored_home_r10` (0.042), `strength_gd_diff_r20` (0.021). 5 features tactical_xg en top 15 SHAP.

**Fase 4 — Calibración**: reliability diagrams por clase. Modelo YA está bien calibrado sin post-hoc — isotonic regression lo empeora (0.9831 → 1.1862). Confirmación de que LightGBM Poisson + Dixon-Coles produce probabilidades naturalmente calibradas. **NO se aplica corrección**.

**Fase 5 — Backtest 3 temporadas + Monte Carlo**:

| Temporada | N | Modelo LL | Pinnacle LL | Ratio |
|---|---|---|---|---|
| 2022-23 | 1826 | 0.9859 | 0.9749 | 1.0113 |
| 2023-24 | 1752 | 0.9793 | 0.9525 | 1.0282 |
| 2024-25 | 1752 | 0.9818 | 0.9606 | 1.0221 |
| **Promedio** | — | — | — | **1.0205** |

Monte Carlo LaLiga 24/25 (5000 sims): Madrid 48.4% campeón (real: 1º), Barça 44.6% (real: 2º), Atlético 4.7%/75.3% top-4 (real: 3º). Descensos: Valladolid 75.9%, Las Palmas 55.2% (ambos descendieron).

**Fase 6 — Dashboard Streamlit**: `app/dashboard.py`, 4 páginas (Resumen, Predicción partido concreto, Simulación temporada, Backtest). Corre en localhost:8501.

### Roadmap v2 documentado en PLAN.md
- Familia D real (datos jugador granular) — ingesta FBref player_match_stats + Transfermarkt + Sofifa
- Análisis táctico live (tracking data)
- Champions League vía custom league_dict
- Player embeddings (bloqueado por datos jugador)
- Ligas adicionales

## [Unreleased]

### Added (2026-07-24 continued)
- Feature families implemented: A (strength, 19), B (form, 12), C (context, 7), E (tactical, 18, dummy-tested — awaits xG data), F (matchup, 5).
- `src/metrics.py` — reusable proper-scoring-rule metrics module extracted from Phase 1.
- `src/build_features_basic.py` — consolidates A+B+C+F into single Parquet.
- `src/fase2_lgb_basic.py` — Phase 2 preliminary experiment: LightGBM Poisson on 43 basic features.
- `src/ingest_fbref_light.py` — attempted `read_schedule()` bulk ingest (works but does NOT expose xG).
- `src/ingest_fbref_shooting.py` — targeted `read_team_match_stats(stat_type='shooting')` for xG per match (running).

### Key empirical finding — Phase 2 preliminary
LightGBM with 43 features derived only from results (ELO, form, H2H, context) achieves **log-loss 0.9888** on test 2024-25 — **worse than the Phase 1 baseline (0.9859)** with a single feature (ELO diff).

Feature importance shows ELO diff dominates absolutely (17,172 gain vs 2,949 next). Additional features derived from results add noise, not signal, because ELO already integrates that information implicitly.

**Conclusion**: the real value jump requires FBref advanced stats (xG, PPDA, progressive passes). This validates the PLAN's design that Phase 2 without FBref is insufficient. Not a failure — an informative negative result.

### Investigation notes
- `read_schedule()` in soccerdata does NOT include xG despite FBref showing xG in the web schedule table. xG requires `read_team_match_stats(stat_type='shooting')` — much slower (scrapes team-by-team-by-season).
- Serie A failed silently during `read_schedule` bulk ingest — investigation deferred; Football-Data.co.uk covers Serie A fully so no data loss for baseline.

### In progress
- FBref shooting stats ingest (xG per match) — background job.

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
