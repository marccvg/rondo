# Fantasy Football Predictor

**Sistema predictivo Fantasy Premier League (FPL) — sugerencias óptimas de fichajes y gestión de plantilla.**

## Objetivo

Predice puntos fantasy por jugador por jornada + optimiza plantilla de 15 jugadores dado presupuesto £100M FPL.

## Alcance v1

- **Liga**: Fantasy Premier League (FPL oficial) — la más popular globalmente
- **Datos**: 100% FPL API oficial (gratis, sin auth) + FBref stats históricos
- **Predicciones**: puntos próxima jornada + próximas 5 jornadas
- **Optimizador**: elige XI titular + suplentes maximizando puntos esperados

## Arquitectura

```
1. Ingesta datos
   ├─ FPL API: precios, historial jugadores, fixtures, difficulty
   └─ FBref: xG, xA, minutos por jugador y partido

2. Sistema puntuación FPL
   Reglas oficiales aplicadas: goles, asistencias, clean sheets, tarjetas,
   penales, bonus points, playing 60+ minutos

3. Feature engineering
   ├─ Forma últimos 5 partidos jugador
   ├─ xG/xA rolling 5, 10 partidos
   ├─ Fixture difficulty próximas 5 jornadas
   ├─ Minutos rolling (proxy titular)
   ├─ Home/away split
   └─ Precio y ownership

4. Modelo predictivo
   LightGBM entrenado sobre GW1-GWn histórico
   Target: puntos fantasy próxima jornada

5. Optimizador plantilla
   Linear Programming (PuLP):
   - 15 jugadores (2 GK, 5 DEF, 5 MID, 3 FWD)
   - Presupuesto £100M
   - Máximo 3 jugadores por club
   - Maximiza puntos esperados suma XI titular + factor suplentes
```

## Uso

```bash
# Ingesta actualizada
venv/bin/python fantasy/src/ingest_fpl_api.py

# Entrenar modelo
venv/bin/python fantasy/src/train_predictor.py

# Optimizador equipo
venv/bin/python fantasy/src/optimize_team.py --budget 100

# Dashboard interactivo
venv/bin/streamlit run fantasy/app/fantasy_dashboard.py
```

## Datos generados

- `data/raw/fpl_bootstrap.json` — snapshot completo FPL API
- `data/raw/fpl_fixtures.json` — todos los partidos con difficulty
- `data/raw/fpl_players_history.parquet` — historial GW por GW cada jugador
- `data/processed/player_features.parquet` — features engineering para modelo
- `models/fpl_predictor_lgb.txt` — modelo LightGBM entrenado
