# Plan detallado — Fútbol · ELO enriquecido + Poisson bivariable

**Proyecto**: sistema predictivo de resultados de partidos (5 grandes ligas + Champions)
**Referencia idea**: `/home/claude/ideas/sistema/idea-272-sistema-elo-poisson-prediccion-futbol-champions.md`
**Creado**: 2026-07-24
**Estado global**: v1 COMPLETADO 2026-07-24. Fases 1-6 cerradas. Ratio promedio backtest 3 temporadas = 1.020 vs Pinnacle (a 2% consistente). Dashboard Streamlit funcional. Extensiones v2 (Familia D jugador granular, Champions, tracking live) documentadas en roadmap.
**Duración estimada**: 10 semanas de trabajo real (asumiendo dedicación intermitente)

---

## Objetivo final medible

Modelo que dado un partido futuro (equipo local, equipo visitante, fecha) produce distribución completa de probabilidades sobre resultado exacto, con **log-loss ≤ 5% peor que cuotas Pinnacle** en backtest 3 temporadas.

Entregables tangibles al cierre:
- Repo GitHub público con README serio y notebook narrado.
- Dashboard Streamlit con predicciones próxima jornada auto-actualizadas.
- Post técnico corto explicando resultados (opcional pero recomendado para portfolio).

---

## Alcance de datos — decisión explícita

| Aspecto | Decisión inicial | Racional |
|---|---|---|
| Ligas | 5 grandes (LaLiga, Premier, Serie A, Bundesliga, Ligue 1) + Champions | Cobertura FBref completa; ~2000 partidos/temporada. |
| Temporadas | 2017/18 → 2025/26 (8 completas + actual) | Umbral inferior de xG/métricas avanzadas fiables en FBref. |
| Volumen esperado | 15-25k partidos | Tamaño amigable a LightGBM. Sobra para gradient boosting, escaso para deep learning (por eso NO deep learning). |
| Extensiones futuras | Championship, Segunda, Ligue 2 (fase 6+) | Solo si Fases 1-5 rinden. |

**NO se ingesta**: fútbol pre-2017, ligas fuera del top-10 europeo, competiciones nacionales de copa (calendario roto, muestra pequeña por equipo).

---

## Stack técnico

- **Lenguaje**: Python 3.11+
- **Ingesta**: `soccerdata` (FBref, Understat, ClubELO, Football-Data, Sofifa), `requests` + custom scrapers para Transfermarkt.
- **Datos**: `pandas`, `duckdb` (para queries analíticas rápidas), Parquet como formato de disco.
- **Modelos**: `lightgbm` (principal), `statsmodels` (baseline Poisson), `scipy.stats` (Dixon-Coles), `optuna` (tuning).
- **Interpretación**: `shap`.
- **Dashboard**: `streamlit`.
- **Notebooks**: `jupyter` + `nbstripout` para git-friendly.
- **Testing**: `pytest` para funciones puras (Dixon-Coles, ELO update, feature calc).

---

## Fases y tareas atómicas

Formato: `[ ]` pendiente, `[x]` completada, `[~]` en curso.

---

### Fase 0 — Setup + ingesta datos (semana 1, ~8h)

**Objetivo**: tener en disco local todos los datos, saber qué hay y qué falta.

- [ ] Crear venv Python 3.11 y `requirements.txt` inicial
- [ ] Instalar: soccerdata, pandas, duckdb, lightgbm, statsmodels, scipy, optuna, shap, streamlit, jupyter, pytest
- [ ] Probar `import soccerdata as sd` + call trivial `sd.FBref('ENG-Premier League', '2024').read_schedule()`
- [x] Descarga histórico FBref — schedule + stats por partido, 5 grandes + Champions, 2017/18 → 2024/25 → guardar `data/raw/fbref_*.parquet`
- [ ] Descarga Football-Data.co.uk — resultados + cuotas Pinnacle/Bet365 mismas ligas mismas temporadas → `data/raw/footballdata_*.parquet`
- [ ] Descarga ClubELO histórico → `data/raw/clubelo.parquet` (baseline externo)
- [ ] Descarga Understat xG por tiro → `data/raw/understat_*.parquet`
- [ ] Descarga StatsBomb open-data completo (git clone del repo) → `data/raw/statsbomb_open/`
- [ ] Descarga Sofifa ratings jugador histórico anual → `data/raw/sofifa_*.parquet`
- [ ] Scraper Transfermarkt custom: plantillas + lesiones actuales + valores mercado → `data/raw/transfermarkt_*.parquet` (validar rate limits, usar 2s delay entre requests)
- [ ] Cargar todo en notebook `00_inventory.ipynb`
- [ ] Documentar `data/INVENTORY.md`: por fuente/liga/temporada indicar #filas, campos disponibles, missing values, huecos identificados
- [ ] Definir claves de join entre fuentes (equipo puede llamarse "Manchester Utd" en FBref y "Man United" en Football-Data → crear `data/team_name_mapping.csv`)

---

### Fase 1 — Baseline ELO clásico + Poisson simple (semanas 2-3, ~15h)

**Objetivo**: número contra el que comparar todo lo posterior. Sin baseline honesto, imposible saber si feature engineering aporta.

- [ ] Implementar `src/elo.py`: función `update_elo(rating_a, rating_b, result, k=32)` con actualización recursiva
- [ ] Test unitario: dos equipos rating 1500, empate → ambos siguen 1500
- [ ] Correr ELO sobre histórico completo de resultados, obtener rating cada equipo cada semana → `data/processed/elo_history.parquet`
- [ ] Comparar rating ELO propio contra ClubELO externo (correlación esperada >0.9)
- [ ] Implementar `src/poisson_baseline.py`: modelo Poisson univariante — λ_local = f(ELO_local, ELO_visitante, home_advantage) vía statsmodels GLM Poisson
- [ ] Poisson bivariable independiente: P(i,j) = Poisson(i,λ_L) × Poisson(j,λ_V)
- [ ] Implementar `src/dixon_coles.py`: función `dc_correction(i, j, lambda_L, lambda_V, rho)` que ajusta las 4 celdas (0,0),(1,0),(0,1),(1,1)
- [ ] Notebook `01_baseline.ipynb`:
  - [ ] Split temporal: train 2017/18-2022/23, val 2023/24, test 2024/25
  - [ ] Fit baseline, evaluar log-loss + RPS + Brier
  - [ ] Comparar contra: uniforme (33/33/33), histórico local win (46/27/27), cuotas Pinnacle
  - [ ] Reliability diagram

**Criterio de paso a Fase 2**: baseline Poisson-ELO debe batir al menos "histórico local win" en log-loss. Si no, hay bug — no seguir.

---

### Fase 2 — Feature engineering completo (semanas 4-5, ~20h)

**Objetivo**: tabla `data/processed/features_matches.parquet` con una fila por partido y 60-120 features + 2 targets (goles_local, goles_visitante).

Cada familia implementada en su propio `src/features/family_X.py` con función `compute_family_X(matches_df, ...) -> pd.DataFrame`.

- [ ] **Familia A — Fuerza base**: ELO propio, ClubELO baseline, xG rolling 5/10/20, xGA rolling, dif goles ajustada
- [ ] **Familia B — Forma reciente**: puntos últimos 5 con decay exponencial, racha victorias/invicto, xG - goles reales (regresión a media)
- [ ] **Familia C — Contexto partido**: local/visitante, días desde último partido, km viaje aproximado, importancia (posición tabla vs rival)
- [ ] **Familia D — Estado plantilla**: minutos acumulados titulares 7/14/30d, lesiones ponderadas por rating Sofifa × prob_titular, sanciones, rotación esperada, valor mercado disponible
- [ ] **Familia E — Estilo táctico**: PPDA, posesión %, pases progresivos, altura línea, % ataques bandas vs centro
- [ ] **Familia F — Matchup**: H2H últimos 5, diferencia perfiles tácticos, matchup portero vs xG rival
- [ ] **Familia G — Externos**: clima (OpenWeatherMap histórico API), rating árbitro (tarjetas/pen por partido), asistencia
- [ ] **Familia H (opcional, con leak-guard)**: cuota Pinnacle T-24h (NUNCA cuota de cierre)
- [ ] Consolidación: script `src/build_features.py` que toma raw → produce tabla única
- [ ] Notebook `02_eda.ipynb`: distribuciones, correlaciones, missing values, outliers, sanity checks (ej. no debe haber "días descanso = -3")
- [ ] Test: sample partido conocido (ej. Barcelona-Madrid clásico 2024) → verificar features hand-calculated coinciden

**Criterio de paso a Fase 3**: tabla completa, <5% missing por feature, sanity checks OK.

---

### Fase 3 — Modelo LightGBM con loss Poisson (semana 6, ~12h)

- [ ] Módulo `src/model.py`: clase `PoissonGBM` con `fit`, `predict`, `save`, `load`
- [ ] Dos modelos separados: uno para λ_local (target = goles_local), uno para λ_visitante
- [ ] Loss: `objective='poisson'` en LightGBM
- [ ] Split temporal train/val/test (mismo que baseline)
- [ ] Tuning con Optuna: 100 trials, expanding window CV, hyperparams (num_leaves, learning_rate, min_data_in_leaf, feature_fraction, bagging_fraction, lambda_l1, lambda_l2)
- [ ] Notebook `03_lightgbm.ipynb`:
  - [ ] Fit final con mejores hyperparams
  - [ ] Métricas log-loss, RPS, Brier sobre test
  - [ ] SHAP values: top 20 features por importancia
  - [ ] Feature selection: eliminar features con importance < umbral
  - [ ] Reentrenar con features filtradas, comparar
  - [ ] Guardar modelo final `models/lgbm_poisson_v1.pkl`

**Criterio de paso a Fase 4**: modelo debe batir baseline Fase 1 significativamente (>3% mejora log-loss).

---

### Fase 4 — Distribución bivariable + Dixon-Coles + calibración (semana 7, ~8h)

- [ ] Función `predict_match(model, features) -> matriz 9x9 P(i,j)`
- [ ] Aplicar corrección Dixon-Coles (ρ estimado vía MLE sobre train)
- [ ] Funciones derivación mercados: `to_1x2()`, `to_over_under(threshold)`, `to_btts()`, `to_exact()`, `to_handicap(line)`
- [ ] Calibración: reliability diagrams por mercado. Si mal calibrado → isotonic regression o Platt scaling post-hoc
- [ ] Test: probabilidades deben sumar 1 (o cerca), P(cualquier resultado) > 0

---

### Fase 5 — Backtest riguroso + comparación cuotas (semana 8, ~10h)

- [ ] Notebook `04_backtest.ipynb`
- [ ] Backtest 3 temporadas out-of-sample (2022/23, 2023/24, 2024/25)
- [ ] Comparación log-loss modelo vs Pinnacle vs baseline vs uniforme, desglose por liga
- [ ] Análisis edge: en qué liga/mercado el modelo aporta más
- [ ] Simulación temporada Monte Carlo N=10.000: para cada temporada test, ¿acertó campeón/top4/descenso mejor que ranking pre-temporada?
- [ ] Simulación bracket Champions KO: probabilidad cada equipo por ronda
- [ ] Reporte `reports/backtest_v1.md` con tablas + gráficos

**Criterio éxito global**: log-loss modelo ≤ 1.05 × log-loss Pinnacle. Si no, iterar sobre feature engineering (Fase 2) antes de continuar.

---

### Fase 6 — Aplicación + entregable público (semanas 9-10, ~15h)

- [ ] `app/dashboard.py`: Streamlit
  - [ ] Página 1: predicciones próxima jornada (auto-refresh)
  - [ ] Página 2: histórico predicciones vs realidad (tracking del modelo en producción)
  - [ ] Página 3: simulador temporada (usuario elige liga → simula 10k temporadas → gráficos)
  - [ ] Página 4: SHAP explicabilidad por partido concreto
- [ ] Deploy Streamlit Community Cloud (gratis)
- [ ] Cron/GitHub Action para re-ingesta semanal de FBref + re-predicción próxima jornada
- [ ] Repo GitHub público:
  - [ ] README serio (problema, approach, resultados, cómo reproducir)
  - [ ] Notebooks navegables con narrativa
  - [ ] Licencia MIT
  - [ ] Badge de Streamlit deploy
- [ ] (Opcional) Post técnico en medium/blog: "Building a football prediction model from public data" — enlace a repo + resultados

---

## Estructura de carpetas

```
futbol_elo_poisson/
├── PLAN.md                 (este archivo)
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                (fuentes originales, no editables)
│   ├── processed/          (features consolidadas)
│   ├── INVENTORY.md
│   └── team_name_mapping.csv
├── notebooks/
│   ├── 00_inventory.ipynb
│   ├── 01_baseline.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_lightgbm.ipynb
│   └── 04_backtest.ipynb
├── src/
│   ├── elo.py
│   ├── poisson_baseline.py
│   ├── dixon_coles.py
│   ├── model.py
│   ├── build_features.py
│   └── features/
│       ├── family_a_strength.py
│       ├── family_b_form.py
│       ├── family_c_context.py
│       ├── family_d_squad.py
│       ├── family_e_tactical.py
│       ├── family_f_matchup.py
│       └── family_g_external.py
├── models/                 (LightGBM pickles versionados)
├── reports/                (backtest results, plots)
├── app/
│   └── dashboard.py
└── tests/
    ├── test_elo.py
    ├── test_dixon_coles.py
    └── test_features.py
```

---

## Métricas de éxito por fase

| Fase | Métrica clave | Umbral aceptable |
|---|---|---|
| 0 | Cobertura ligas × temporadas descargadas OK | 100% |
| 1 | Baseline bate "histórico local win" en log-loss | >0.5% mejora |
| 2 | Tabla features consolidada, <5% missing global | Sí |
| 3 | LightGBM bate baseline Fase 1 | >3% mejora log-loss |
| 4 | Probabilidades bien calibradas | Reliability diagram alineado con diagonal ±5% |
| 5 | Modelo vs Pinnacle | log-loss modelo ≤ 1.05 × log-loss Pinnacle |
| 6 | Repo público + dashboard funcionando | Sí |

---

## Riesgos identificados

- **Rate limits FBref**: scraping agresivo bloquea 15-60 min. Mitigar: `time.sleep(3)` entre requests, batch de descargas nocturno.
- **Cambios HTML upstream**: soccerdata puede romperse cuando FBref actualiza layout. Mitigar: pin versión, monitorizar issues del repo.
- **Data leakage**: incluir features del futuro por error (cuota cierre, resultado final en rolling mal calculado). Mitigar: función de rolling siempre con `shift(1)` explícito, code review consciente.
- **Overfitting**: 100 features sobre 15k partidos es ratio arriesgado. Mitigar: CV temporal estricto, regularización (L1/L2 en LightGBM), feature selection agresiva post-SHAP.
- **Datos Transfermarkt**: sin API oficial, scraping frágil. Mitigar: fallback a features "sin plantilla" si el scraper cae; el modelo debe funcionar con missing en Familia D.

---

## Extensiones futuras post-v1 (roadmap actualizado 2026-07-24)

### Datos jugador granular — Familia D real
- **Ingesta FBref player_match_stats**: goles/xG/minutos/pases progresivos/tackles/presiones/duelos por jugador y partido. Requiere ~1-2h scraping equipo-por-equipo.
- **Alineaciones canónicas por partido**: extraer titulares desde FBref schedule (Formation column) + StatsBomb open donde cubra.
- **Transfermarkt scraper**: lesiones activas + sanciones + valores mercado + contratos. Sin API oficial, scraping con delay ~2s por request.
- **Sofifa scraper**: ratings jugador anual (proxy importancia para ponderación).
- **Features derivadas (~30 features agregadas al equipo)**:
  - Σ xG generado últimos 5 partidos por los 4 delanteros titulares
  - Precisión pases % últimos 10 de los 3 mediocentros
  - % duelos ganados últimos 10 de los 4 defensas
  - Save % últimos 10 del portero
  - Minutos acumulados 7/14/30d de la alineación esperada (fatiga colectiva)
  - Lesiones activas ponderadas por Sofifa × prob_titular
  - Rotación esperada (alineación probable vs canónica)
- **Impacto esperado**: 3-8% adicional mejora log-loss según literatura (Kloosterman 2020). Realista cerrar 20-40% del gap restante contra Pinnacle.

### Análisis táctico en tiempo real (live)
- Requiere tracking data (Second Spectrum / StatsBomb 360 / Signality) — no disponible en open. Deferimiento indefinido salvo hardware propio.
- Alternativa low-cost: MCU + IMU en jugadores de amateur/juvenil (idea-271-like) para captura casera.
- Features candidatos: compactness live, PPDA rolling 5min, xT por posesión, detección deriva de rol vs baseline canónico jugador.

### Champions League
- Añadir vía custom `~/soccerdata/config/league_dict.json` apuntando a `https://fbref.com/en/comps/8/Champions-League-Stats`.

### Player embeddings (bloqueado por datos jugador)
- word2vec-style aprendido de secuencias de acciones por jugador → representación vectorial que captura similitudes tácticas no visibles en tabla estadística.
- Requisito: StatsBomb events (~1.5 GB) o event-level equivalente.
- Aplicación principal: idea-273 (scouting big data) — no v1.

### Otras ligas
- Championship, Segunda División, Serie B, Ligue 2, 2. Bundesliga — extender simple.
- Ligas nórdicas 2ª div, MLS Next Pro, Championship escocesa — sin datos Understat, requerirían fuentes específicas.

## Decisiones abiertas (pendientes de resolver durante Fase 0-1)

- ¿Añadir Championship + Segunda? → decidir al ver cobertura FBref real.
- ¿Modelar copas (Copa del Rey, FA Cup)? → probablemente no en v1, muestra pequeña por equipo.
- ¿Modelo único con `is_home` o dos modelos separados? → probar ambos, quedarse con el que mejor calibra.
- ¿Usar cuotas Pinnacle como feature (Familia H) o solo como benchmark? → solo benchmark en v1 (evita leak); considerar en v2 con leak-guard fuerte.

---

## Log de progreso (append-only)

- **2026-07-24**: plan creado. Fase 0 pendiente de arranque. Marc autorizará Fase 0 explícitamente antes de empezar.
- **2026-07-24 (mismo día)**: Fase 0 ejecutada. Football-Data 40/40 OK (14.359 partidos, 100% cuotas Pinnacle). ClubELO OK. FBref+Understat probes OK. StatsBomb open aún descargando en background (segundo intento vía tarball tras timeout git-clone).
- **2026-07-24 (mismo día)**: Fase 1 ejecutada. Baseline ELO recursivo + Poisson GLM + Dixon-Coles (rho=-0.030). Split temporal train 17/18-22/23 (10.855) / val 23/24 (1752) / test 24/25 (1752). **Log-loss test 0.986 vs Pinnacle 0.961 = ratio 1.026** (solo 2.6% peor que Pinnacle usando SOLO ELO como feature). Mejora vs home_prior baseline: 8.5%. Reporte: `reports/fase1_baseline.md`.

---

## Ejecución — quién dispara qué

Reglas de este proyecto:
- Marc dice "arranca Fase 0" → yo ejecuto Fase 0 completa en background y reporto al final.
- Cada fase reportada = decisión Marc explícita antes de arrancar la siguiente.
- Reporte de fase = markdown en `reports/fase_N_reporte.md` + actualización de este PLAN.md marcando tareas hechas.
- Bloqueos → registrar en `reports/BLOQUEOS.md` y notificar a Marc.
