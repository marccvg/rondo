# pipeline/refresh.py — auto-actualizador del proyecto

**Ejecución típica** (desde la raíz del proyecto):

```bash
# Actualizar datos + ELO (default, ~1 min):
venv/bin/python pipeline/refresh.py

# Actualizar + predecir todas las competiciones:
venv/bin/python pipeline/refresh.py --predict all

# Solo una competición:
venv/bin/python pipeline/refresh.py --predict champions
venv/bin/python pipeline/refresh.py --predict laliga
venv/bin/python pipeline/refresh.py --predict premier

# Con xG Understat (lento, ~10 min por Selenium arrancando):
venv/bin/python pipeline/refresh.py --with-understat

# Solo recomputar ELO sin bajar (útil si ya bajaste antes):
venv/bin/python pipeline/refresh.py --no-download
```

## Qué hace cada fase

1. **Fase 1 — Football-Data.co.uk**: descarga incremental. Para temporadas ya cerradas: skip si existe. Para la **temporada actual**: siempre re-descargar (jornadas nuevas cada semana).
2. **Fase 2 — Understat (opcional)**: descarga xG por partido. Selenium/Chrome, lento.
3. **Fase 3 — Reconstruir matches_all.parquet**: consolida los CSVs en tabla única.
4. **Fase 4 — Recomputar ELO**: recorre TODO el histórico, guarda ratings finales + snapshot con timestamp.
5. **Fase 5 — Comparar vs snapshot anterior**: reporta cambios >5 puntos ELO por equipo.
6. **Fase 6 — Predicción**: opcional, regenera las predicciones para la próxima temporada.

## Cadencia recomendada

- **Cada 2-4 semanas** durante temporada activa (agosto-mayo): captura resultados nuevos, evolución de forma.
- **Una vez en verano** (junio-julio): capta la temporada recién terminada completa.
- **Al empezar temporada** (agosto): con `--predict all` para tener predicciones frescas.

## Archivos que se actualizan

- `data/raw/footballdata/*.csv` — nuevos CSVs descargados
- `data/processed/matches_all.parquet` — tabla consolidada
- `data/processed/matches_with_elo.parquet` — con ELO añadido
- `data/processed/final_elo_ratings.json` — últimos ratings
- `data/snapshots/elo_snapshot_YYYYMMDD_HHMMSS.json` — histórico (permite comparativa temporal)
- `mini/mini_final_elo.json` — para compatibilidad con learning/

## No toca

- `models/` — modelos LightGBM entrenados en v1 (usa `--retrain` en futuro para re-entrenar)
- `src/` — código de v1 congelado
- `tests/`, `app/` — igual
