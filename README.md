# futbol_elo_poisson

**Sistema predictivo de fútbol: 23.537 partidos cronológicos (2017-2026), ELO cross-league calibrado con coeficiente UEFA + Poisson bivariable + Monte Carlo.**

## Estado del proyecto

| Componente | Estado |
|---|---|
| Dataset | **23.537 partidos**, 9 competiciones, 9 temporadas, 762 equipos |
| ELO cross-league | K adaptativo por competición (UCL/UEL/UECL) + K por liga derivado UEFA coefficient |
| Modelo predictivo | Poisson GLM + Dixon-Coles + Monte Carlo (5.000 sims/predicción) |
| Fuentes | 100% públicas verificables (Football-Data, openfootball, Wikipedia, Understat) |
| Reproducibilidad | 1 comando (`pipeline/refresh.py`) actualiza todo |
| Trazabilidad | `data/DATA_PROVENANCE.md` documenta cada fuente |

## Competiciones cubiertas

| Competición | Partidos | Temporadas | Fuente |
|---|---|---|---|
| Premier League | 3.420 | 2017-2026 | Football-Data.co.uk |
| LaLiga | 3.420 | 2017-2026 | Football-Data.co.uk |
| Serie A | 3.420 | 2017-2026 | Football-Data.co.uk |
| Bundesliga | 2.754 | 2017-2026 | Football-Data.co.uk |
| Ligue 1 | 3.097 | 2017-2026 | Football-Data.co.uk |
| Portugal Primeira | 2.142 | 2017-2026 (huecos 19/20, 20/21) | Football-Data.co.uk |
| Champions League | 1.247 | 2017-2026 | openfootball GitHub |
| Europa League | 1.939 | 2017-2026 | Wikipedia scraper |
| Conference League | 2.098 | 2021-2026 (competición creada 21/22) | Wikipedia scraper |

## Arquitectura del modelo

### 1. Datos → 2. ELO recursivo → 3. Poisson + Dixon-Coles → 4. Monte Carlo

**K adaptativo por competición** (elección jerárquica basada en importancia):
- Champions League: K=50 (×2.5 base)
- Europa League: K=40 (×2.0)
- Conference League: K=30 (×1.5)
- Ligas domésticas: K derivado del **UEFA country coefficient 5-year rolling** calculado sobre nuestros propios datos

**K ligas domésticas actuales** (temporadas 21/22-25/26):
```
England:  20.00  (top UEFA coefficient)
Spain:    17.52
Portugal: 15.96
Germany:  15.43
Italy:    15.19
France:   13.13
```

### Fórmulas

**Update ELO recursivo** (Arpad Elo 1960 + adaptaciones fútbol):
```
Expected(A vs B) = 1 / (1 + 10^((elo_B − (elo_A + home_adv))/400))
Delta = K × goal_diff_multiplier × (Actual − Expected)
```

**Poisson bivariable + Dixon-Coles** (Dixon-Coles 1997):
```
log(λ_home) = α_H + β_H · elo_diff
log(λ_away) = α_A + β_A · elo_diff
P(i,j) = Poisson(i,λ_H) × Poisson(j,λ_A) × τ_DC(i,j,λ_H,λ_A,ρ)
```

## Cómo ejecutar

### Requisitos
- Python 3.11+
- Dependencias: `pip install -r requirements.txt`

### Comando único de actualización
```bash
# Baja datos nuevos + recompute ELO + regenera predicciones:
venv/bin/python pipeline/refresh.py --predict all
```

### Ejecuciones específicas
```bash
# Sólo datos + ELO (sin predicciones):
venv/bin/python pipeline/refresh.py

# Recomputar coeficiente UEFA + K derivado:
venv/bin/python src/uefa_coefficient.py

# Recomputar ELO UEFA-calibrado:
venv/bin/python src/elo_uefa_calibrated.py

# Predicciones 5 competiciones 2026/27:
venv/bin/python learning/predict_all_5_v2_realistic_pools.py

# Aprendizaje paso a paso (script pedagógico):
venv/bin/python learning/walkthrough.py
# O modo automático sin pausas:
WALKTHROUGH_AUTO=1 venv/bin/python learning/walkthrough.py
```

## Últimas predicciones 2026/27

Ver `reports/` o ejecutar el predictor. Top:

**Champions League**: Bayern Munich 23.7%, PSG 23.5%, Arsenal 17.2%
**LaLiga**: Barcelona 53%, Real Madrid 43%
**Premier**: Arsenal 96.9% (⚠️ ver limitaciones en NEW_MODEL_DOCS.md)

## Estructura del repo

```
futbol_elo_poisson/
├── README.md, PLAN.md, CHANGELOG.md, LICENSE
├── requirements.txt
├── data/
│   ├── DATA_PROVENANCE.md      ← auditoría de fuentes (crítico)
│   ├── raw/                     ← datos crudos scrapeados (gitignored)
│   └── processed/               ← consolidados + ratings ELO (gitignored)
├── src/
│   ├── data_loader.py           ← consolida CSVs Football-Data
│   ├── elo.py                   ← ELO base clásico
│   ├── elo_multi.py             ← ELO cross-league K constante
│   ├── elo_adaptive_k.py        ← ELO con K UEFA cups adaptativo
│   ├── elo_uefa_calibrated.py   ← ELO con K derivado UEFA coefficient  ★ actual
│   ├── uefa_coefficient.py      ← calcula UEFA country coef sobre nuestros datos
│   ├── dixon_coles.py           ← corrección bivariable Poisson
│   ├── merge_all_competitions.py← unifica dominios + UEFA cups cronológico
│   ├── ingest_champions.py      ← Champions League (openfootball)
│   ├── ingest_uefa_cups_full.py ← Europa + Conference (Wikipedia)
│   └── features/                ← engineering features (Fase 2 antigua)
├── pipeline/
│   ├── refresh.py               ← auto-actualizador principal
│   ├── PLAN_EUROPEAN_CUPS.md    ← plan integración UEFA cups
│   └── diagnose_*.py            ← herramientas diagnóstico
├── learning/
│   ├── walkthrough.py           ← recorrido pedagógico paso-a-paso
│   ├── predict_all_5_v2_*.py    ← predictor 5 competiciones 2026/27
│   └── predict_*.py             ← predictores específicos
├── mini/                        ← versión pedagógica original 8 scripts
├── app/dashboard.py             ← Streamlit dashboard
└── tests/                       ← pytest tests
```

## Filosofía y limitaciones conocidas

**Filosofía**: datos verificables, cálculos deterministas, transparencia total.

**Limitaciones estructurales** (documentadas):
1. **Modelo ELO simple** → sobre-reacciona a rachas cortas (Arsenal Premier 26/27 sale 96%, artefacto de dominio reciente + Champions histórica)
2. **Sin regression to mean** → equipos que colapsan una temporada no rebotan automáticamente
3. **Datos gratis** → techo log-loss ~2% por encima de Pinnacle (que tiene info privada)
4. **Pool UEFA cups 2026/27 aproximado** → composición real no conocida hasta agosto 2026

**Ver `NEW_MODEL_DOCS.md` para detalles arquitectónicos**.

## Licencia
MIT
