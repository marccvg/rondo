# Mini-proyecto — mismo algoritmo, mínimos datos

Objetivo pedagógico: entender el pipeline completo con **solo 4 datos por partido**:
`home_team, away_team, home_goals, away_goals`. Nada más. Sin xG, sin cuotas, sin plantillas.

Cada script imprime **números intermedios** para poder seguir el algoritmo con ejemplos concretos.

## Cómo ejecutar todo en orden

```bash
cd /home/Projects/futbol_elo_poisson
venv/bin/python mini/01_carga_datos.py
venv/bin/python mini/02_elo_paso_a_paso.py
venv/bin/python mini/03_poisson_glm.py
venv/bin/python mini/04_lightgbm.py
venv/bin/python mini/05_dixon_coles.py
venv/bin/python mini/06_predice_un_partido.py
venv/bin/python mini/07_simula_temporada.py
```

## Cada paso muestra

- **01** — qué contienen los 14.359 partidos, ejemplos concretos, primeras filas
- **02** — ELO recursivo: ver un partido específico, el rating antes/después, la fórmula aplicada con números
- **03** — Poisson GLM: los 2 coeficientes aprendidos + traducción a "goles esperados"
- **04** — LightGBM: features derivadas (solo 4 sencillas) + primeros splits del árbol + gains
- **05** — Dixon-Coles: la matriz 9×9 antes y después de la corrección, celdas específicas
- **06** — dado (Real Madrid, Barcelona) hoy: qué predice el modelo, número a número
- **07** — Monte Carlo 10.000 temporadas → probabilidad campeón por equipo

## Métrica final

Mismo test 2024-25 que el pipeline grande. Comparación directa:

- Baseline grande (Fase 1): log-loss 0.9859
- Mini: X.XXXX (a rellenar tras 04)

Debe salir similar — porque la Fase 1 grande usa esencialmente los mismos inputs, solo que
más elegante en la implementación.
