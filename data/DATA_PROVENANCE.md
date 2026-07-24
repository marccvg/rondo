# Procedencia de datos — auditoría de fuentes

Documento este archivo para garantizar que **ningún dato del proyecto es inventado**. Todos vienen de fuentes públicas verificables.

## Fuentes por tipo de dato

| Dato | Fuente | Verificable en |
|---|---|---|
| Resultados 5 grandes ligas + Portugal | Football-Data.co.uk | `https://www.football-data.co.uk/mmz4281/{YYYY}/{XX}.csv` |
| Cuotas Pinnacle (para benchmark) | Football-Data.co.uk (columnas PSCH/PSCD/PSCA) | Ídem |
| Resultados Champions League | openfootball/champions-league (GitHub) | `https://raw.githubusercontent.com/openfootball/champions-league/master/{YYYY-YY}/cl.txt` |
| Resultados Europa League | Wikipedia (parseado con pandas.read_html) | `https://en.wikipedia.org/wiki/{YYYY–YY}_UEFA_Europa_League_*` |
| Resultados Conference League | Wikipedia (parseado con pandas.read_html) | `https://en.wikipedia.org/wiki/{YYYY–YY}_UEFA_Conference_League_*` |
| ClubELO snapshot (referencia externa) | api.clubelo.com | `http://api.clubelo.com/{YYYY-MM-DD}` |
| xG por partido (Understat) | soccerdata → Understat.com | scrapper oficial soccerdata |

## Valores DERIVADOS / calculados (no scrapeados)

Estos NO son datos reales — son valores computados por nuestro modelo:

- **ELO ratings** (todos): resultado de aplicar la fórmula ELO clásica sobre los resultados reales scrapeados
- **ELO inicial 1500**: valor por defecto matemático (standard Arpad Elo 1960) para equipos sin histórico previo
- **Coeficientes Poisson GLM**: aprendidos por statsmodels.GLM sobre los datos reales
- **ρ Dixon-Coles = -0.030**: aprendido por grid search sobre datos reales

## Valores APROXIMADOS

- **Fecha exacta de partidos UEFA (Europa/Conference)**: usamos "noviembre del año inicial de temporada" como fecha aproximada. Los RESULTADOS son reales de Wikipedia, la fecha exacta día-a-día no se extrae para simplificar. Impacto: mínimo — el ELO se ejecuta ordenado por fecha, y agrupar todos los partidos UEFA de una temporada en noviembre no cambia el rating final (solo la trayectoria intermedia).

## Auditoría de reproducibilidad

Cualquier persona puede reproducir el dataset ejecutando:
```bash
venv/bin/python src/ingest_champions.py          # baja de openfootball GitHub
venv/bin/python src/ingest_uefa_cups_full.py     # baja de Wikipedia
venv/bin/python pipeline/refresh.py              # baja Football-Data
```

Y comparar con el archivo original en la fuente. **Todo trazable.**

## Compromiso

No se genera ningún resultado ficticio. Si un partido no está en la fuente, no está en el dataset.
