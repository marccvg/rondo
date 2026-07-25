"""
WALKTHROUGH v2 — algoritmo completo paso a paso.

Versión actualizada con:
- 23.537 partidos (5 grandes + Portugal + Champions + Europa + Conference)
- K adaptativo por competición
- Coeficiente UEFA para K por liga
- Ejemplos concretos con números REALES post-25/26

Ejecutar:
  venv/bin/python learning/walkthrough_v2.py           # con pausas
  WALKTHROUGH_AUTO=1 venv/bin/python learning/walkthrough_v2.py  # sin pausas
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path("/home/Projects/futbol_elo_poisson")
sys.path.insert(0, str(ROOT / "src"))

import warnings
warnings.filterwarnings("ignore")

AUTO = os.environ.get("WALKTHROUGH_AUTO", "0") == "1"
_orig_input = input
def input(prompt=""):
    if AUTO:
        print(prompt)
        return ""
    return _orig_input(prompt)


def separator(t=""):
    print("\n" + "═" * 78)
    print(f"  {t}")
    print("═" * 78)


def subsep(t=""):
    print("\n" + "─" * 62)
    print(f"  {t}")
    print("─" * 62)


# ═══════════════════════════════════════════════════════════════════════
# INTRODUCCIÓN
# ═══════════════════════════════════════════════════════════════════════

separator("WALKTHROUGH v2 — FÚTBOL ELO CROSS-LEAGUE CALIBRADO")

print("""
Recorrido paso a paso del modelo predictivo actual (post-integración UEFA cups):

  1️⃣  Datos: 23.537 partidos, 9 competiciones, 762 equipos
  2️⃣  ELO recursivo — fórmula clásica Arpad Elo
  3️⃣  K adaptativo — la clave del cross-league calibration
  4️⃣  UEFA coefficient — K por liga derivado de datos reales
  5️⃣  Poisson GLM — aprender goles esperados
  6️⃣  Dixon-Coles — corrección de resultados bajos
  7️⃣  Predicción de un partido real con números
""")

input("Pulsa ENTER para empezar...")


# ═══════════════════════════════════════════════════════════════════════
# 1. DATOS
# ═══════════════════════════════════════════════════════════════════════

separator("1️⃣  DATOS — inventario completo")

df = pd.read_parquet(ROOT / "data/processed/matches_all_multi.parquet")
df = df.sort_values("date").reset_index(drop=True)

print(f"✅ {len(df):,} partidos cronológicos")
print(f"   Rango: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"   Total equipos únicos: {len(set(df['home_team']) | set(df['away_team']))}")

pivot = df.groupby(["season", "competition"]).size().unstack(fill_value=0)
print(f"\nDistribución por competición y temporada:")
print(pivot.to_string())

print(f"""
Fuentes (100% verificables en data/DATA_PROVENANCE.md):
  - Football-Data.co.uk    → 6 ligas domésticas (18.253 partidos)
  - openfootball GitHub    → Champions League (1.247)
  - Wikipedia              → Europa League + Conference (4.037)
""")

input("Pulsa ENTER para pasar a ELO...")


# ═══════════════════════════════════════════════════════════════════════
# 2. ELO RECURSIVO CLÁSICO
# ═══════════════════════════════════════════════════════════════════════

separator("2️⃣  ELO RECURSIVO — Arpad Elo 1960 aplicado a fútbol")

print("""
La fórmula que aplicamos partido a partido:

  ┌─────────────────────────────────────────────────────────────────┐
  │ Expected(A vs B) = 1 / (1 + 10^((elo_B − (elo_A + 65))/400))   │
  │                                                                  │
  │ Delta = K × gd_mult × (Actual − Expected)                       │
  │                                                                  │
  │ nuevo_elo_A = elo_A + delta                                     │
  │ nuevo_elo_B = elo_B − delta         (zero-sum)                  │
  └─────────────────────────────────────────────────────────────────┘

  Actual: 1 (gana A), 0.5 (empate), 0 (gana B)
  home_adv = 65 puntos (ventaja de local)
  gd_mult: 1 gol→×1.0, 2→×1.5, ≥3→×(11+gd)/8
  K: VARIABLE por competición (siguiente sección)

Todos los equipos empiezan en 1500.
""")

input("Pulsa ENTER para ver K adaptativo...")


# ═══════════════════════════════════════════════════════════════════════
# 3. K ADAPTATIVO
# ═══════════════════════════════════════════════════════════════════════

separator("3️⃣  K ADAPTATIVO — la clave del cross-league calibration")

print("""
K = "learning rate" del ELO. Cuánto se mueve el rating tras cada partido.

📊 K por competición (nivel de importancia):
    Champions League:   50   ← partidos cross-liga top, mucha info
    Europa League:      40
    Conference League:  30
    Ligas domésticas:   VARIABLE (siguiente sección)

Rationale: un partido Champions te dice MUCHÍSIMO más sobre nivel del equipo
que un partido de liga contra rival aplastantemente inferior.

Sin K adaptativo (K=20 fijo):
  → Sp Lisbon acumula MUCHO ELO ganando en Portugal (238 partidos × +2 = +476)
  → Sus derrotas Champions solo restan poco (42 × -5 = -210)
  → Balance neto: Sp Lisbon SIEMPRE arriba
  → Modelo dice Sp Lisbon > Liverpool  (irreal)

Con K adaptativo (UCL=50):
  → Sus derrotas Champions restan más (42 × -12 = -504)
  → Balance real: Sp Lisbon baja hacia realidad
""")

input("Pulsa ENTER para ver UEFA coefficient...")


# ═══════════════════════════════════════════════════════════════════════
# 4. UEFA COEFFICIENT
# ═══════════════════════════════════════════════════════════════════════

separator("4️⃣  UEFA COEFFICIENT — K por liga desde datos reales")

print("""
Además del K por competición, cada LIGA DOMÉSTICA tiene su propio K basado
en cuánto rinde en Europa (formula oficial UEFA aplicada a nuestros datos).

Fórmula UEFA:
  - Victoria en competición UEFA: +2 puntos país
  - Empate: +1 punto
  - Total país / equipos participantes = coef temporada
  - Coef 5-year = suma últimas 5 temporadas

Calculado sobre nuestros 5.284 partidos UEFA cups:
""")

coef_data = json.load(open(ROOT / "data/processed/uefa_country_k.json"))
ranked = sorted(coef_data.items(), key=lambda x: -x[1])
print(f"{'País':<15} {'K liga':>10}  {'% vs top':>10}")
print("─" * 40)
max_k = ranked[0][1]
for c, k in ranked:
    print(f"{c:<15} {k:>10.2f}  {k/max_k*100:>9.1f}%")

print(f"""
Interpretación:
  England K=20 → top: sus victorias domésticas cuentan al 100%
  France K=13.1 → Ligue 1 rinde peor en Europa → menos peso a Ligue 1 doméstico
  → PSG deja de inflarse por dominio absoluto Ligue 1
""")

input("Pulsa ENTER para Poisson GLM...")


# ═══════════════════════════════════════════════════════════════════════
# 5. POISSON GLM
# ═══════════════════════════════════════════════════════════════════════

separator("5️⃣  POISSON GLM — de ELO a goles esperados")

print("""
Con ELO calibrado, entrenamos regresión Poisson para predecir goles:

  log(λ_home) = α_H + β_H × elo_diff
  log(λ_away) = α_A + β_A × elo_diff

Coeficientes aprendidos sobre partidos históricos:
  α_H = 0.3883,  β_H = 0.001842  (goles esperados home)
  α_A = 0.1763,  β_A = -0.001824 (goles esperados away)

Interpretación con números reales (equipos igualados, elo_diff=0):
  λ_home = exp(0.3883) = 1.47 goles esperados local
  λ_away = exp(0.1763) = 1.19 goles esperados visitante
  Diferencia natural = home advantage aprendido automáticamente
""")

input("Pulsa ENTER para Dixon-Coles...")


# ═══════════════════════════════════════════════════════════════════════
# 6. DIXON-COLES
# ═══════════════════════════════════════════════════════════════════════

separator("6️⃣  DIXON-COLES — corrección de dependencia bajo-anotación")

print("""
Poisson independiente asume que goles home y away son independientes.
La realidad: NO lo son. Partidos 0-0, 1-0, 0-1, 1-1 son más frecuentes.

Razón intuitiva: el que va perdiendo ataca más, el que va ganando defiende.

Dixon-Coles 1997 corrige esas 4 celdas con factor τ(i,j,λH,λA,ρ):
  τ(0,0) = 1 - λH · λA · ρ
  τ(1,0) = 1 + λA · ρ
  τ(0,1) = 1 + λH · ρ
  τ(1,1) = 1 - ρ

Nuestro modelo aprende ρ = -0.030 (grid search sobre training)
""")

def dc_tau(i, j, lh, la, rho):
    if i == 0 and j == 0: return 1 - lh * la * rho
    if i == 0 and j == 1: return 1 + lh * rho
    if i == 1 and j == 0: return 1 + la * rho
    if i == 1 and j == 1: return 1 - rho
    return 1

def score_matrix(lh, la, rho=-0.03, size=9):
    m = np.zeros((size, size))
    ph = poisson.pmf(np.arange(size), lh)
    pa = poisson.pmf(np.arange(size), la)
    for i in range(size):
        for j in range(size):
            m[i,j] = ph[i] * pa[j] * dc_tau(i, j, lh, la, rho)
    return m / m.sum()

mat = score_matrix(1.5, 1.0)
print("\nEjemplo λ_home=1.5, λ_away=1.0 (matriz 6×6 primeras celdas):")
header = "  ".join(f"A={j}" for j in range(6))
print(f"        {header}")
for i in range(6):
    row = " ".join(f"{mat[i,j]:.3f}" for j in range(6))
    print(f"  H={i}  {row}")

input("\nPulsa ENTER para predicción real...")


# ═══════════════════════════════════════════════════════════════════════
# 7. PREDICCIÓN COMPLETA
# ═══════════════════════════════════════════════════════════════════════

separator("7️⃣  PREDICCIÓN COMPLETA — Real Madrid vs Barcelona")

elos = json.load(open(ROOT / "data/processed/final_elo_ratings_uefa_calibrated.json"))
h = "Real Madrid"
a = "Barcelona"
eh = elos.get(h, 1500)
ea = elos.get(a, 1500)
diff = eh - ea

print(f"\n1️⃣  Ratings ELO actuales (UEFA-calibrado):")
print(f"    {h}: {eh:.1f}")
print(f"    {a}: {ea:.1f}")
print(f"    elo_diff: {diff:+.1f}")

# Poisson coef aprendidos
α_H, β_H = 0.3883, 0.001842
α_A, β_A = 0.1763, -0.001824

log_lh = α_H + β_H * diff
log_la = α_A + β_A * diff
lh = np.exp(log_lh)
la = np.exp(log_la)

print(f"\n2️⃣  Predicción goles esperados (Poisson GLM):")
print(f"    log(λ_home) = {α_H} + {β_H} × {diff:+.1f} = {log_lh:.4f}")
print(f"    log(λ_away) = {α_A} + {β_A} × {diff:+.1f} = {log_la:.4f}")
print(f"    λ_home = {lh:.3f}")
print(f"    λ_away = {la:.3f}")

mat = score_matrix(lh, la, rho=-0.03)
n = mat.shape[0]
ph = sum(mat[i,j] for i in range(n) for j in range(n) if i > j)
pd_ = sum(mat[i,i] for i in range(n))
pa = sum(mat[i,j] for i in range(n) for j in range(n) if i < j)

print(f"\n3️⃣  Matriz probabilidad (Poisson + DC):")
header5 = "  ".join(f"A={j}" for j in range(5))
print(f"        {header5}")
for i in range(5):
    row = " ".join(f"{mat[i,j]:.3f}" for j in range(5))
    print(f"  H={i}  {row}")

print(f"\n4️⃣  Probabilidades finales:")
print(f"    P({h} gana):   {ph*100:.1f}%")
print(f"    P(Empate):     {pd_*100:.1f}%")
print(f"    P({a} gana):   {pa*100:.1f}%")

# Top 5 resultados
flat = [(i,j,mat[i,j]) for i in range(n) for j in range(n)]
flat.sort(key=lambda x: -x[2])
print(f"\n5️⃣  Top 5 resultados más probables:")
for i, j, p in flat[:5]:
    print(f"    {h} {i}-{j} {a}  →  {p*100:.2f}%")


separator("✅ WALKTHROUGH v2 TERMINADO")

print("""
Has recorrido el pipeline actualizado con:
  ✅ 23.537 partidos cronológicos (6 ligas + 3 UEFA cups)
  ✅ ELO con K adaptativo por competición
  ✅ K por liga derivado del UEFA coefficient
  ✅ Poisson GLM + Dixon-Coles

Rankings finales guardados en:
  data/processed/final_elo_ratings_uefa_calibrated.json

Para predicciones 2026/27:
  venv/bin/python learning/predict_all_5_v2_realistic_pools.py

Estado del proyecto: maduro y funcional dentro de los límites de datos gratis
+ arquitectura ELO clásica. Ver NEW_MODEL_DOCS.md para más detalle.
""")
