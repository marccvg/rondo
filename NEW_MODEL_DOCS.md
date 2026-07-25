# Arquitectura del modelo — versión actual

## Evolución del modelo a lo largo del proyecto

| Iteración | Descripción | Log-loss test 24/25 | Diferencial vs Pinnacle |
|---|---|---|---|
| 1 baseline | ELO clásico K=20, 5 grandes ligas | 0.9859 | +2.6% |
| 2 features engineering | +43 features derivados solo resultados | 0.9888 | +2.9% (peor, dilution) |
| 2 FULL + xG | +xG de Understat (61 features) | 0.9818 | +2.2% |
| 3 Optuna tuned | Hyperparam search LightGBM | 0.9831 | +2.3% |
| 4 calibración | Isotonic post-hoc | 0.9831 | +2.3% (isotonic empeora, no aplicado) |
| **Actual** | **+Champions + Europa + Conference + Portugal + K adaptativo + UEFA coefficient** | **~0.98 (estimado)** | **~2.2%** |

## K adaptativo — decisiones tomadas

### K por competición UEFA (jerárquico por importancia)
```
Champions League:   50  (×2.5 sobre base 20)
Europa League:      40  (×2.0)
Conference League:  30  (×1.5)
```

**Racionalidad**: partidos cross-liga aportan más información que domésticos (calibración cross-liga real).

### K por liga doméstica (derivado UEFA country coefficient)

Calculado con **fórmula UEFA oficial** aplicada sobre nuestros 5.284 partidos UEFA cups:

Fórmula simplificada:
- Victoria en competición UEFA: +2 puntos país
- Empate: +1 punto
- Puntos totales país / equipos participantes = coeficiente temporada
- Coef 5-year = suma últimas 5 temporadas

**Coeficientes calculados** (temporadas 21/22 - 25/26):
```
England:  36.33  → K liga = 20.00
Spain:    31.82  → K liga = 17.52
Portugal: 29.00  → K liga = 15.96
Germany:  28.03  → K liga = 15.43
Italy:    27.59  → K liga = 15.19
France:   23.86  → K liga = 13.13
```

**Fórmula K**: `K_liga = 20 × (coef_liga / max_coef)`

### Home advantage
```
home_adv = 65 puntos ELO
```
Convención estándar (ClubELO). No hay valor oficial UEFA, pero 65 es común en literatura.

### Goal difference multiplier
```
1 gol diferencia:  ×1.0
2 goles:           ×1.5
≥3 goles:          ×(11 + |gd|) / 8
```
Adaptación empírica para fútbol (FiveThirtyEight-style). Recompensa más las victorias claras.

## Fórmulas completas

### Update ELO
```python
def update_elo(elo_home, elo_away, home_goals, away_goals, K, home_adv=65):
    # Probabilidad esperada (curva logística Arpad Elo original)
    expected_home = 1 / (1 + 10**((elo_away - (elo_home + home_adv)) / 400))
    
    # Resultado real
    if home_goals > away_goals: actual = 1.0
    elif home_goals < away_goals: actual = 0.0
    else: actual = 0.5
    
    # Multiplicador por margen de goles
    gd = abs(home_goals - away_goals)
    if gd <= 1: mult = 1.0
    elif gd == 2: mult = 1.5
    else: mult = (11 + gd) / 8
    
    # Delta (K adaptativo según competición del partido)
    delta = K × mult × (actual - expected_home)
    
    return elo_home + delta, elo_away - delta
```

### Poisson bivariable + Dixon-Coles
```python
# Regresión Poisson entrenada sobre partidos históricos
log_λ_home = α_H + β_H × elo_diff
log_λ_away = α_A + β_A × elo_diff

# Matriz probabilidad (9×9)
for i in range(9):
    for j in range(9):
        P(i,j) = Poisson(i, λ_home) × Poisson(j, λ_away) × τ_DC(i,j,λ_home,λ_away,ρ)

# Dixon-Coles corrige 4 celdas bajas:
τ_DC(0,0) = 1 − λ_home × λ_away × ρ
τ_DC(1,0) = 1 + λ_away × ρ
τ_DC(0,1) = 1 + λ_home × ρ
τ_DC(1,1) = 1 − ρ
τ_DC(i,j) = 1  para el resto
```

Coeficientes aprendidos:
- α_H = 0.3883 (intercepto goles home)
- β_H = 0.001842 (slope elo_diff)
- α_A = 0.1763 (intercepto goles away)
- β_A = -0.001824 (slope elo_diff)
- ρ = -0.030 (Dixon-Coles óptimo, grid search)

## Auditoría de fuentes

Ver `data/DATA_PROVENANCE.md`. Resumen:

| Fuente | Datos | Verificable en |
|---|---|---|
| Football-Data.co.uk | Resultados 6 ligas domésticas | URLs `mmz4281/{YYYY}/{XX}.csv` |
| openfootball GitHub | Champions League 2015-2026 | `github.com/openfootball/champions-league` |
| Wikipedia | Europa League + Conference | `en.wikipedia.org/wiki/{YYYY-YY}_UEFA_*` |
| Understat (via soccerdata) | xG por partido | scrapper `soccerdata` |
| ClubELO API | Snapshot rating externo (referencia) | `api.clubelo.com/{date}` |

## Limitaciones conocidas

1. **Sobre-reacción a rachas recientes**: modelo ELO simple no tiene "regression to mean". Un equipo que gana muchas Champions consecutivas dispara su ELO más de lo que quizá refleja fuerza real.

2. **Canibalización interna liga**: en ligas competitivas (Premier), equipos top se pelean entre sí → ELO se estanca. En ligas dominadas (Portugal), top absorbe ELO de débiles → se infla. K UEFA coefficient corrige parcialmente pero no del todo.

3. **Volumen desproporcionado doméstico/UEFA**: 18.253 partidos domésticos vs 5.284 UEFA cups. La calibración cross-liga tiene menos peso relativo del ideal.

4. **Pool UEFA 2026/27 aproximado**: la lista real de clasificados UEFA no se conoce hasta agosto 2026 (tras qualifying). Pool actual es aproximación por clasificaciones 25/26.

5. **Modelo Poisson simple (1 feature)** para predicciones futuras: LightGBM con 61 features no es aplicable a partidos futuros sin conocer las features específicas del partido.

6. **Techo intrínseco del fútbol**: hay componente irreducible de aleatoriedad. Ningún modelo alcanza log-loss 0. Pinnacle (con info privada) está en 0.96. Nosotros ~0.98. El gap absoluto es ~1-2% inevitable con datos públicos.

## Reproducibilidad

Cualquiera puede recrear el dataset ejecutando:
```bash
python src/ingest_champions.py              # Champions desde openfootball
python src/ingest_uefa_cups_full.py         # Europa+Conference desde Wikipedia
python pipeline/refresh.py                  # Football-Data + Understat
python src/merge_all_competitions.py        # Consolidar cronológico
python src/uefa_coefficient.py              # Calcular UEFA coef
python src/elo_uefa_calibrated.py           # ELO final
```

Todos los resultados son deterministas (0 aleatoriedad excepto Monte Carlo con seed fija).
