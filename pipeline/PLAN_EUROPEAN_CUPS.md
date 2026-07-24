# Plan — integración completa competiciones europeas UEFA

**Objetivo**: consolidar en el modelo TODOS los partidos de las 3 competiciones UEFA de clubes desde 2017/18, cronológicamente ordenados junto a los domésticos, y re-entrenar ELO desde cero.

## Estado inicial (antes del plan)

| Competición | Cobertura previa |
|---|---|
| Champions League | ✅ 2017-2026 (openfootball, 1.247 partidos) |
| Europa League | ⚠️ Solo 24/25+25/26 (Wikipedia Swiss parser, 288 partidos) |
| Conference League | ❌ 0 partidos |

## Estado objetivo (post-plan)

| Competición | Cobertura objetivo | Fuente | Fase |
|---|---|---|---|
| Champions League | 17/18 → 25/26 | openfootball | ✅ hecho |
| **Europa League** | **17/18 → 25/26 (9 temp)** | Wikipedia parser mejorado | **fase 1 del plan** |
| **Conference League** | **21/22 → 25/26 (5 temp, todas las que existen)** | Wikipedia | **fase 2 del plan** |

## Fase 1 — Europa League histórica completa (17/18 a 23/24)

Wikipedia estructura Europa League por temporada:
- 17/18 - 23/24: formato "group stage + knockout stage"
- 24/25 - 25/26: formato "league phase Swiss + knockout"

Retos del parser:
1. **Tablas cruzadas** en group stage clásico (matriz N×N donde celda = resultado)
2. **Tablas de fixture** en knockout stage (columnas ida/vuelta)
3. **Nombres equipos** varían entre fuentes (Wikipedia usa "Bayer Leverkusen", openfootball "Bayer 04 Leverkusen")

Enfoque:
- Detectar formato "cross" por columnas: si columnas contienen nombres de equipos (mismos que filas), es formato cruzado
- Extraer resultados celda a celda excluyendo diagonal
- Fecha aproximada: usar mediana de temporada Europa (noviembre-abril)

## Fase 2 — Conference League

Existe desde 21/22. 5 temporadas. Formato:
- 21/22 - 23/24: group stage + knockout
- 24/25 - 25/26: league phase Swiss + knockout

Mismo parser que Europa League.

## Fase 3 — Consolidación cronológica + re-ELO

1. Unificar en `matches_all_multi.parquet`:
   - 6 competiciones domésticas (5 grandes + Portugal)
   - 3 competiciones UEFA (Champions + Europa + Conference)
2. Ordenar por fecha
3. Re-ejecutar ELO recursivo sobre TODO
4. Comparar rankings resultantes vs versión anterior + ClubELO

## Fase 4 — Verificación

- Liverpool debería subir de #13 a top 10 (o cercano)
- Ingleses en general deberían acercarse a ClubELO (+100-200 puntos actualmente en gap)
- Portugueses deberían estabilizarse en su posición realista
- Bayern probablemente cae ligeramente (competencia europea a mano)

## Estimaciones honestas

| Fase | Trabajo estimado | Impacto esperado en Liverpool |
|---|---|---|
| Fase 1 | 2-3h (Wikipedia parser custom para cross tables) | +40 a +80 |
| Fase 2 | 1h (mismo parser, menos temporadas) | +5 (juega Europa no Conference) |
| Fase 3 | 15 min (script existente) | inmediato tras 1+2 |
| Fase 4 | 15 min (comparativa) | validación |
| **Total** | **~4-5h** | **Liverpool sube ~50-100 puntos → top 10** |

## Métrica de éxito

**Ratio nuestro Liverpool / ClubELO Liverpool ≥ 0.95** (actualmente 1715/1910 = 0.90). Objetivo: 1810/1910 = 0.95 tras integración completa.
