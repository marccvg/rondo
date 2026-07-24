# Carpeta learning/ — versión pedagógica

**No modifica el proyecto v1** — es una copia autoexplicativa para aprender el algoritmo paso a paso viendo NÚMEROS.

## Scripts

- **`walkthrough.py`** — recorrido completo del algoritmo con prints ultra-explicativos:
  - Introducción: qué datos tenemos y por qué
  - ELO recursivo: fórmula + 3 partidos con números
  - Poisson GLM: coeficientes aprendidos + interpretación
  - LightGBM: los gradientes iteración a iteración (cada 10 rondas), la estructura del árbol 1
  - Dixon-Coles: matriz antes y después
  - Predicción de un partido real de arriba a abajo
- **`predict_champions_2025_26.py`** — predice Champions próxima temporada:
  - 32 top equipos europeos por ELO
  - Sorteo aleatorio de 8 grupos
  - Simulación Monte Carlo de grupos → clasificados
  - Octavos aleatorios cross grupos (1º vs 2º de otro grupo)
  - Cuartos, semis, final
  - Estimación probabilidad campeón

## Cómo ejecutar

```bash
cd /home/Projects/futbol_elo_poisson
venv/bin/python learning/walkthrough.py            # ~30-60 seg, mucho output
venv/bin/python learning/predict_champions_2025_26.py  # ~1-2 min
```

## Diferencias con v1

- No toca `src/`, `models/`, `data/`, `mini/`, `app/`.
- Usa modelos ya entrenados de `models/` (solo lectura).
- Datos leídos de `data/processed/` (solo lectura).
- Prints y comentarios abundantes, orientados a comprensión no a producción.
