"""
WALKTHROUGH — algoritmo completo paso a paso con números.

NO modifica nada del proyecto v1. Solo lee datos y modelos.
Ejecutar: venv/bin/python learning/walkthrough.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson

# Modo automático: set WALKTHROUGH_AUTO=1 para saltar las pausas input()
AUTO_MODE = os.environ.get("WALKTHROUGH_AUTO", "0") == "1"
_original_input = input
def input(prompt=""):
    if AUTO_MODE:
        print(prompt)
        return ""
    return _original_input(prompt)

ROOT = Path("/home/Projects/futbol_elo_poisson")
sys.path.insert(0, str(ROOT / "src"))

# Silenciar warnings ruidosos de pandas
import warnings
warnings.filterwarnings("ignore")


def separador(titulo=""):
    print("\n" + "═" * 78)
    print(f"  {titulo}")
    print("═" * 78)


def subseparador(titulo=""):
    print("\n" + "─" * 60)
    print(f"  {titulo}")
    print("─" * 60)


# ═══════════════════════════════════════════════════════════════════════
# INTRODUCCIÓN
# ═══════════════════════════════════════════════════════════════════════

separador("WALKTHROUGH — FÚTBOL ELO + POISSON + LIGHTGBM")

print("""
Este script recorre PASO A PASO todo el algoritmo del proyecto v1.
Su propósito es DIDÁCTICO: verás los NÚMEROS reales que produce cada
etapa, para entender el "por qué" y no solo el "cómo".

Estructura:
  1️⃣  Con qué datos trabajamos (fuente, cobertura, ejemplos)
  2️⃣  ELO recursivo — la fórmula aplicada a partidos reales
  3️⃣  Poisson GLM — regresión lineal en el espacio log(λ)
  4️⃣  LightGBM — gradient boosting con LOSS Poisson, iteración a iteración
  5️⃣  Dixon-Coles — corrección de dependencia bajo-anotación
  6️⃣  Predicción real de un partido, número a número
""")

input("Pulsa ENTER para empezar...")


# ═══════════════════════════════════════════════════════════════════════
# 1. DATOS
# ═══════════════════════════════════════════════════════════════════════

separador("1️⃣  CON QUÉ DATOS TRABAJAMOS")

print("""
Los datos provienen de 3 fuentes públicas y gratuitas:

  📊 Football-Data.co.uk
      - CSVs por temporada de las 5 grandes ligas europeas
      - Contiene: fecha, equipo local, equipo visitante, goles, resultado
      - IMPORTANTE: incluye cuotas de cierre de Pinnacle Sports
        (usadas SÓLO como benchmark, nunca como feature)

  📊 Understat (via soccerdata)
      - Datos por partido con xG (expected goals) del propio modelo Understat
      - Cobertura: 6 grandes ligas europeas desde 2014

  📊 ClubELO
      - Ratings ELO externos para benchmark contra nuestro ELO propio

Vamos a cargar los partidos y ver qué contienen.
""")

df = pd.read_parquet(ROOT / "data" / "processed" / "matches_all.parquet")
print(f"✅ Cargados {len(df):,} partidos")
print(f"   Rango temporal: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"   Ligas: {sorted(df['league'].unique())}")
print(f"   Total equipos únicos: {len(set(df['home_team']) | set(df['away_team']))}")

print(f"\nPrimeras 3 filas:")
print(df.head(3)[["date", "league", "home_team", "away_team", "home_goals", "away_goals", "result", "pinnacle_home", "pinnacle_draw", "pinnacle_away"]].to_string(index=False))

print(f"\nDistribución de resultados (H = local gana, D = empate, A = visitante gana):")
print((df["result"].value_counts(normalize=True).round(3) * 100).to_string())

print(f"\n💡 Nota importante:")
print(f"   El fútbol tiene un sesgo natural pro-local: ~43% victorias local, 32% visitante, 25% empate.")
print(f"   Cualquier modelo ha de superar este 'home prior' (predecir siempre local con 43%).")

input("\nPulsa ENTER para pasar a ELO...")


# ═══════════════════════════════════════════════════════════════════════
# 2. ELO RECURSIVO
# ═══════════════════════════════════════════════════════════════════════

separador("2️⃣  ELO RECURSIVO — LA FÓRMULA CON NÚMEROS")

print("""
El sistema ELO asigna a cada equipo un RATING numérico que actualiza
después de cada partido. Fórmula:

  ┌─────────────────────────────────────────────────────────────────┐
  │  prob_esperada_home = 1 / (1 + 10^((elo_away - (elo_home + H)) / 400))     │
  │  delta = K × gd_mult × (resultado_real - prob_esperada)         │
  │  nuevo_elo_home = elo_home + delta                              │
  │  nuevo_elo_away = elo_away - delta                              │
  └─────────────────────────────────────────────────────────────────┘

  H (home advantage) = 65 puntos ELO
  K (learning rate)  = 20
  gd_mult (multiplicador por diferencia goles):
    1 gol de dif → ×1.0
    2 goles      → ×1.5
    ≥3 goles     → ×(11 + gd)/8
  resultado_real: 1 (gana local), 0.5 (empate), 0 (gana visitante)

Todos los equipos empiezan con ELO = 1500 (neutral).
""")

INITIAL, K, HOME_ADV = 1500.0, 20.0, 65.0

def expected(rh, ra):
    return 1.0 / (1.0 + 10 ** ((ra - (rh + HOME_ADV)) / 400.0))

def gd_mult(gd):
    g = abs(gd)
    if g <= 1: return 1.0
    if g == 2: return 1.5
    return (11.0 + g) / 8.0

df_sorted = df.sort_values("date").reset_index(drop=True)
ratings = {}
def get(t): return ratings.get(t, INITIAL)

print("Voy a mostrarte 3 partidos aplicando la fórmula paso a paso:\n")

for idx in [0, 100, 5000]:
    row = df_sorted.iloc[idx]
    subseparador(f"Partido histórico #{idx+1}: {row['home_team']} {row['home_goals']}-{row['away_goals']} {row['away_team']} ({row['date'].date()})")
    rh_before = get(row["home_team"])
    ra_before = get(row["away_team"])
    print(f"  ELO ANTES:")
    print(f"    {row['home_team']:20s} = {rh_before:.1f}")
    print(f"    {row['away_team']:20s} = {ra_before:.1f}")

    exp = expected(rh_before, ra_before)
    print(f"\n  📐 Prob esperada de que gane el local:")
    print(f"     = 1 / (1 + 10^(({ra_before:.0f} - ({rh_before:.0f} + 65)) / 400))")
    print(f"     = 1 / (1 + 10^({(ra_before - (rh_before + 65))/400:.4f}))")
    print(f"     = {exp:.4f}  ({exp*100:.1f}%)")

    if row["home_goals"] > row["away_goals"]:
        actual = 1.0; icon = "🟢 gana local"
    elif row["home_goals"] < row["away_goals"]:
        actual = 0.0; icon = "🔴 gana visitante"
    else:
        actual = 0.5; icon = "🟡 empate"
    print(f"\n  Resultado real: {icon} → actual = {actual}")

    gd = row["home_goals"] - row["away_goals"]
    m = gd_mult(gd)
    print(f"  gd_mult({gd:+d}) = {m}")

    delta = K * m * (actual - exp)
    print(f"\n  📊 delta = 20 × {m} × ({actual} - {exp:.4f}) = {delta:+.2f}")

    rh_new = rh_before + delta
    ra_new = ra_before - delta
    print(f"\n  ELO DESPUÉS:")
    print(f"    {row['home_team']:20s} = {rh_before:.1f} + {delta:+.2f} = {rh_new:.1f}")
    print(f"    {row['away_team']:20s} = {ra_before:.1f} - {delta:+.2f} = {ra_new:.1f}")

    # aplicar cambios y avanzar
    ratings[row["home_team"]] = rh_new
    ratings[row["away_team"]] = ra_new

    # avanzar los partidos intermedios sin mostrarlos
    if idx < 5000:
        for j in range(idx+1, min(idx+100 if idx == 0 else 5000, len(df_sorted))):
            if j >= 5000: break
            r2 = df_sorted.iloc[j]
            rh, ra = get(r2["home_team"]), get(r2["away_team"])
            e = expected(rh, ra)
            a = 1.0 if r2["home_goals"] > r2["away_goals"] else (0.0 if r2["home_goals"] < r2["away_goals"] else 0.5)
            d = K * gd_mult(r2["home_goals"] - r2["away_goals"]) * (a - e)
            ratings[r2["home_team"]] = rh + d
            ratings[r2["away_team"]] = ra - d

# Correr resto
for j in range(5001, len(df_sorted)):
    r2 = df_sorted.iloc[j]
    rh, ra = get(r2["home_team"]), get(r2["away_team"])
    e = expected(rh, ra)
    a = 1.0 if r2["home_goals"] > r2["away_goals"] else (0.0 if r2["home_goals"] < r2["away_goals"] else 0.5)
    d = K * gd_mult(r2["home_goals"] - r2["away_goals"]) * (a - e)
    ratings[r2["home_team"]] = rh + d
    ratings[r2["away_team"]] = ra - d

subseparador(f"Top 15 equipos por ELO tras {len(df_sorted):,} partidos")
top = sorted(ratings.items(), key=lambda kv: -kv[1])[:15]
for i, (t, r) in enumerate(top, 1):
    print(f"  {i:2d}. {t:25s} {r:7.1f}")

print(f"\n💡 Observa cómo el ELO 'redistribuye' desde 1500 inicial.")
print(f"   Los top se colocan entre 1700-1870, los peores bajo 1300.")
print(f"   La 'masa' total del sistema se conserva (zero-sum en cada update).")

input("\nPulsa ENTER para pasar a Poisson GLM...")


# ═══════════════════════════════════════════════════════════════════════
# 3. POISSON GLM
# ═══════════════════════════════════════════════════════════════════════

separador("3️⃣  POISSON GLM — REGRESIÓN LINEAL EN log(λ)")

print("""
El objetivo es predecir el número de GOLES que marcará cada equipo en un partido.

¿Por qué Poisson? Los goles son:
  - Números enteros ≥ 0
  - Eventos raros con tasa media λ conocida
  - La distribución que mejor se ajusta empíricamente

  P(y goles | λ) = e^(-λ) · λ^y / y!

Queremos aprender λ como función lineal de nuestros features:

  log(λ) = w₀ + w₁·x₁ + w₂·x₂ + ...

Empezaremos con UN solo feature: elo_diff = elo_home - elo_away.

  log(λ_home) = α_H + β_H · elo_diff
  log(λ_away) = α_A + β_A · elo_diff

Los coeficientes α y β se aprenden por MÁXIMA VEROSIMILITUD.
Statsmodels usa el algoritmo IRLS (Iteratively Reweighted Least Squares).

Vamos a entrenar sobre train (17/18-22/23) y ver los coeficientes.
""")

# Reconstruir elo pre-partido en df
ratings2 = {}
elo_h, elo_a = [], []
for _, r in df_sorted.iterrows():
    rh, ra = ratings2.get(r["home_team"], INITIAL), ratings2.get(r["away_team"], INITIAL)
    elo_h.append(rh); elo_a.append(ra)
    e = expected(rh, ra)
    a = 1.0 if r["home_goals"] > r["away_goals"] else (0.0 if r["home_goals"] < r["away_goals"] else 0.5)
    d = K * gd_mult(r["home_goals"] - r["away_goals"]) * (a - e)
    ratings2[r["home_team"]] = rh + d
    ratings2[r["away_team"]] = ra - d
df_sorted["elo_home"] = elo_h
df_sorted["elo_away"] = elo_a
df_sorted["elo_diff"] = df_sorted["elo_home"] - df_sorted["elo_away"]

train = df_sorted[df_sorted["date"] < "2023-07-01"].copy()

X = sm.add_constant(train[["elo_diff"]])
mh = sm.GLM(train["home_goals"].astype(float), X, family=sm.families.Poisson()).fit()
ma = sm.GLM(train["away_goals"].astype(float), X, family=sm.families.Poisson()).fit()

print(f"Ajustado sobre {len(train):,} partidos train.\n")
print(f"📐 log(λ_home) = {mh.params['const']:.4f} + {mh.params['elo_diff']:.6f} · elo_diff")
print(f"📐 log(λ_away) = {ma.params['const']:.4f} + {ma.params['elo_diff']:.6f} · elo_diff")

print(f"""
INTERPRETACIÓN CON NÚMEROS CONCRETOS:

  Equipos IGUALADOS (elo_diff = 0):
    λ_home = exp({mh.params['const']:.4f}) = {np.exp(mh.params['const']):.3f} goles esperados
    λ_away = exp({ma.params['const']:.4f}) = {np.exp(ma.params['const']):.3f} goles esperados
    Diferencia natural = ventaja del local (aprendida automáticamente por los interceptos)

  Home 200 ELO SUPERIOR (Real Madrid vs equipo medio):
    λ_home = exp({mh.params['const']:.4f} + {mh.params['elo_diff']:.6f} × 200) = {np.exp(mh.params['const'] + mh.params['elo_diff']*200):.3f}
    λ_away = exp({ma.params['const']:.4f} + {ma.params['elo_diff']:.6f} × 200) = {np.exp(ma.params['const'] + ma.params['elo_diff']*200):.3f}

  Home 200 ELO INFERIOR:
    λ_home = {np.exp(mh.params['const'] + mh.params['elo_diff']*(-200)):.3f}
    λ_away = {np.exp(ma.params['const'] + ma.params['elo_diff']*(-200)):.3f}

💡 β positivo en home y negativo en away = signo esperado.
   Un equipo mejor marca más y recibe menos.
""")

input("Pulsa ENTER para ver LightGBM entrenar iteración a iteración...")


# ═══════════════════════════════════════════════════════════════════════
# 4. LIGHTGBM — GRADIENT BOOSTING PASO A PASO
# ═══════════════════════════════════════════════════════════════════════

separador("4️⃣  LIGHTGBM POISSON — GRADIENT BOOSTING VISIBLE")

print("""
LightGBM construye un modelo apilando muchos ÁRBOLES DE DECISIÓN pequeños.
Cada árbol nuevo intenta corregir los errores del anterior siguiendo la
dirección del gradiente negativo de la loss Poisson.

Para la loss Poisson:
   L(y, F) = e^F - y·F      (donde F = log(λ))

Gradiente: ∂L/∂F = e^F - y = λ - y      (predicción menos realidad)
Hessiana:  ∂²L/∂F² = e^F = λ

  F_m(x) = F_{m-1}(x) + η · h_m(x)

donde h_m es el árbol nuevo que se ajusta al gradiente negativo,
y η es learning_rate (velocidad de convergencia).

Vamos a entrenar con feedback CADA 10 iteraciones para ver la loss bajar.
""")

# Features simples
train["fh_gs5"] = 1.5  # dummy, no matter - solo para demo
train["fh_gc5"] = 1.0

# Usamos solo elo_diff para claridad
X_train = train[["elo_diff"]].values
y_train_h = train["home_goals"].astype(float).values

val = df_sorted[(df_sorted["date"] >= "2023-07-01") & (df_sorted["date"] < "2024-07-01")].copy()
X_val = val[["elo_diff"]].values
y_val_h = val["home_goals"].astype(float).values

print(f"Train: {len(X_train):,} partidos, Val: {len(X_val):,} partidos.")
print(f"Feature única: elo_diff. Objetivo: goles_home.\n")

class VerboseLoss:
    def __init__(self):
        self.rows = []
    def __call__(self, env):
        if env.iteration % 20 == 0 or env.iteration == env.end_iteration - 1:
            val_loss = env.evaluation_result_list[0][2]
            self.rows.append((env.iteration, val_loss))
            print(f"  iter {env.iteration:4d}  |  val poisson loss = {val_loss:.5f}")

train_ds = lgb.Dataset(X_train, y_train_h)
val_ds = lgb.Dataset(X_val, y_val_h, reference=train_ds)

params = {
    "objective": "poisson",
    "metric": "poisson",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 50,
    "verbose": -1,
    "seed": 42,
}
print("Entrenando LightGBM (feedback cada 20 iteraciones):\n")
callback = VerboseLoss()
model = lgb.train(params, train_ds, num_boost_round=200,
                  valid_sets=[val_ds], valid_names=["val"],
                  callbacks=[callback, lgb.log_evaluation(0)])

print(f"\n✅ Modelo entrenado con {len(callback.rows)} snapshots reportados.")
print(f"   Loss inicial (~iter 0): {callback.rows[0][1]:.5f}")
print(f"   Loss final    (iter {callback.rows[-1][0]}): {callback.rows[-1][1]:.5f}")
print(f"   Mejora total: {(callback.rows[0][1] - callback.rows[-1][1])/callback.rows[0][1]*100:.2f}%")

print(f"\n💡 Observa cómo la loss BAJA monotónicamente.")
print(f"   Cada árbol nuevo aporta un poco menos que el anterior (rendimiento marginal decreciente).")
print(f"   En un caso real usaríamos early_stopping para parar cuando dejara de mejorar en val.")

subseparador("Estructura del primer árbol")
tree0 = model.dump_model()["tree_info"][0]["tree_structure"]

def print_tree(node, depth=0):
    prefix = "   " * depth
    if "leaf_value" in node:
        print(f"{prefix}HOJA: valor = {node['leaf_value']:+.5f}  (se suma a log(λ))")
    else:
        thr = node["threshold"]
        gain = node["split_gain"]
        print(f"{prefix}SPLIT: elo_diff ≤ {thr:.2f} ? (gain reducción loss = {gain:.2f})")
        print(f"{prefix}├─ Sí (izq):")
        print_tree(node["left_child"], depth + 1)
        print(f"{prefix}└─ No (der):")
        print_tree(node["right_child"], depth + 1)

print_tree(tree0)

print(f"\n💡 Cada nodo divide el espacio de elo_diff en 2 zonas y asigna un valor.")
print(f"   La 'gain' cuantifica cuánto BAJA la loss al hacer ese split — el criterio de elección.")

input("\nPulsa ENTER para pasar a Dixon-Coles...")


# ═══════════════════════════════════════════════════════════════════════
# 5. DIXON-COLES
# ═══════════════════════════════════════════════════════════════════════

separador("5️⃣  DIXON-COLES — CORRECCIÓN DE POISSON BIVARIABLE")

print("""
Con λ_home y λ_away tenemos la distribución de goles esperados de cada equipo.
Asumiendo independencia, la probabilidad de un resultado (i, j) es:

  P(home=i, away=j) = P(home=i | λ_H) × P(away=j | λ_A)
                    = [Poisson(i, λ_H)] × [Poisson(j, λ_A)]

PROBLEMA: en el fútbol REAL, los 4 resultados (0-0, 1-0, 0-1, 1-1) son más
frecuentes que lo que predice Poisson independiente. La razón: los goles
NO son independientes — un equipo que va perdiendo ataca más, uno que va
ganando defiende, etc.

DIXON-COLES (1997) corrige esas 4 celdas con un factor τ:

  τ(0,0) = 1 - λH · λA · ρ
  τ(1,0) = 1 + λA · ρ
  τ(0,1) = 1 + λH · ρ
  τ(1,1) = 1 - ρ
  τ(i,j) = 1  para el resto

ρ (rho) es un parámetro learnable, típicamente entre -0.05 y 0.05.
Nuestro modelo aprendió ρ = -0.030 sobre train.

Vamos a ver la corrección en acción para un partido típico.
""")

def dc_tau(i, j, lh, la, rho):
    if i == 0 and j == 0: return 1.0 - lh * la * rho
    if i == 0 and j == 1: return 1.0 + lh * rho
    if i == 1 and j == 0: return 1.0 + la * rho
    if i == 1 and j == 1: return 1.0 - rho
    return 1.0

def matriz(lh, la, rho, size=9):
    ph = poisson.pmf(np.arange(size), lh)
    pa = poisson.pmf(np.arange(size), la)
    m = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            m[i, j] = ph[i] * pa[j] * dc_tau(i, j, lh, la, rho)
    return m / m.sum()

lh, la = 1.5, 1.0
m_pura = matriz(lh, la, 0.0)
m_dc = matriz(lh, la, -0.03)

print(f"Ejemplo: partido con λ_home = 1.5, λ_away = 1.0\n")
print(f"Matriz POISSON INDEPENDIENTE (sin corrección):")
print(f"        A=0    A=1    A=2    A=3")
for i in range(4):
    row = "  ".join(f"{m_pura[i,j]:.3f}" for j in range(4))
    print(f"  H={i}  {row}")

print(f"\nMatriz CON DIXON-COLES (ρ = -0.03):")
print(f"        A=0    A=1    A=2    A=3")
for i in range(4):
    row = "  ".join(f"{m_dc[i,j]:.3f}" for j in range(4))
    print(f"  H={i}  {row}")

print(f"\nDiferencia en las 4 celdas ajustadas:")
for (i, j) in [(0,0), (1,0), (0,1), (1,1)]:
    diff = m_dc[i,j] - m_pura[i,j]
    pct = diff / m_pura[i,j] * 100
    print(f"  P({i}-{j}): {m_pura[i,j]:.4f} → {m_dc[i,j]:.4f}  ({pct:+.1f}%)")

print(f"\n💡 DC sube P(0-0) y P(1-1) (empates bajos infra-representados por Poisson puro),")
print(f"   compensando bajando ligeramente P(1-0) y P(0-1). Total suma sigue siendo 1.")

input("\nPulsa ENTER para la predicción final de un partido real...")


# ═══════════════════════════════════════════════════════════════════════
# 6. PREDICCIÓN DE UN PARTIDO REAL
# ═══════════════════════════════════════════════════════════════════════

separador("6️⃣  PREDICCIÓN COMPLETA DE UN PARTIDO REAL")

# Real Madrid vs Barcelona con ELO final
final_elo = ratings2  # el mismo que hemos calculado
h, a = "Real Madrid", "Barcelona"
eh, ea = final_elo.get(h, 1500), final_elo.get(a, 1500)
diff = eh - ea

print(f"Partido hipotético: {h} vs {a}\n")
print(f"1️⃣  Ratings ELO actuales:")
print(f"    {h}: {eh:.1f}")
print(f"    {a}: {ea:.1f}")
print(f"    elo_diff = {diff:+.1f}")

log_lh = mh.params['const'] + mh.params['elo_diff'] * diff
log_la = ma.params['const'] + ma.params['elo_diff'] * diff
lh_pred = np.exp(log_lh)
la_pred = np.exp(log_la)

print(f"\n2️⃣  Predicción Poisson GLM:")
print(f"    log(λ_home) = {mh.params['const']:.4f} + {mh.params['elo_diff']:.6f} × {diff:+.1f} = {log_lh:.4f}")
print(f"    log(λ_away) = {ma.params['const']:.4f} + {ma.params['elo_diff']:.6f} × {diff:+.1f} = {log_la:.4f}")
print(f"    λ_home = exp({log_lh:.4f}) = {lh_pred:.3f}  goles esperados")
print(f"    λ_away = exp({log_la:.4f}) = {la_pred:.3f}  goles esperados")

print(f"\n3️⃣  Matriz DC + probabilidades de mercado:")
mat = matriz(lh_pred, la_pred, -0.03)
n = mat.shape[0]
ph = sum(mat[i,j] for i in range(n) for j in range(n) if i>j)
pd_ = sum(mat[i,i] for i in range(n))
pa = sum(mat[i,j] for i in range(n) for j in range(n) if i<j)
over25 = sum(mat[i,j] for i in range(n) for j in range(n) if i+j > 2)

print(f"    P({h} gana)  = {ph*100:.1f}%")
print(f"    P(Empate)             = {pd_*100:.1f}%")
print(f"    P({a} gana) = {pa*100:.1f}%")
print(f"    P(Over 2.5 goles)     = {over25*100:.1f}%")

flat = [(i, j, mat[i, j]) for i in range(n) for j in range(n)]
flat.sort(key=lambda x: -x[2])
print(f"\n4️⃣  Top 5 resultados exactos más probables:")
for i, j, p in flat[:5]:
    print(f"    {h} {i}-{j} {a}  →  {p*100:.2f}%")


separador("✅ WALKTHROUGH TERMINADO")

print("""
Has recorrido el pipeline completo con NÚMEROS reales:

  Datos → ELO recursivo → Poisson GLM → LightGBM iterativo → Dixon-Coles → Predicción

Este es el mismo algoritmo que produce el resultado de v1:
  log-loss 0.9818 en test 24/25 vs 0.9606 de Pinnacle (ratio 1.022).

Con solo elo_diff (1 feature) ya llegamos a 0.9859 (Fase 1).
Con 61 features (5 familias con xG de Understat) mejoramos a 0.9818 (Fase 2 FULL).

El techo alcanzable con datos públicos gratis está prácticamente cubierto.
Para cerrar más gap: datos jugador granular (Familia D — v2 en roadmap).

Próximo script sugerido: learning/predict_champions_2025_26.py
""")
