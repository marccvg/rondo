"""
MINI paso 2 — ELO RECURSIVO explicado con números.

Fórmula:
  prob_esperada_home = 1 / (1 + 10^((elo_away - (elo_home + home_adv)) / 400))
  delta = K * mult_goles * (resultado_real - prob_esperada)
  nuevo_elo_home = elo_home + delta
  nuevo_elo_away = elo_away - delta
"""
import pandas as pd

INITIAL = 1500.0
K = 20.0
HOME_ADV = 65.0

def expected(rh, ra, home_adv=HOME_ADV):
    return 1.0 / (1.0 + 10 ** ((ra - (rh + home_adv)) / 400.0))

def gd_mult(gd):
    g = abs(gd)
    if g <= 1: return 1.0
    if g == 2: return 1.5
    return (11.0 + g) / 8.0

def elo_update(rh, ra, gh, ga):
    if gh > ga: actual = 1.0
    elif gh < ga: actual = 0.0
    else: actual = 0.5
    exp = expected(rh, ra)
    m = gd_mult(gh - ga)
    delta = K * m * (actual - exp)
    return rh + delta, ra - delta, exp, m, delta

df = pd.read_parquet("/home/Projects/futbol_elo_poisson/mini/mini_matches.parquet")
df = df.sort_values("date").reset_index(drop=True)

print("=" * 60)
print("  MINI paso 2 — ELO RECURSIVO PASO A PASO")
print("=" * 60)

# Ejemplo detallado con los 3 primeros partidos
print(f"\nTODOS los equipos empiezan en {INITIAL} (ranking neutral).")
print(f"K = {K}  |  home_adv = {HOME_ADV}  |  gd_mult: 1gol→x1, 2goles→x1.5, ≥3→x(11+gd)/8\n")

ratings = {}
def get(t): return ratings.get(t, INITIAL)

print("─" * 60)
for i in range(5):
    row = df.iloc[i]
    home, away, gh, ga = row["home_team"], row["away_team"], row["home_goals"], row["away_goals"]
    rh_before = get(home)
    ra_before = get(away)
    rh_after, ra_after, exp, m, delta = elo_update(rh_before, ra_before, gh, ga)
    ratings[home] = rh_after
    ratings[away] = ra_after
    print(f"\nPartido {i+1}: {home} {gh}-{ga} {away} ({row['date'].date()})")
    print(f"  ELO ANTES:  {home}={rh_before:.1f}  {away}={ra_before:.1f}")
    print(f"  Prob esperada home = 1 / (1 + 10^(({ra_before:.0f} - ({rh_before:.0f} + {HOME_ADV})) / 400)) = {exp:.3f}")
    actual = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
    print(f"  Actual = {actual}  (H=1, X=0.5, A=0)")
    print(f"  gd_mult({gh-ga}) = {m}")
    print(f"  delta = {K} * {m} * ({actual} - {exp:.3f}) = {delta:+.2f}")
    print(f"  ELO DESPUÉS: {home}={rh_after:.1f}  {away}={ra_after:.1f}")

print("\n" + "─" * 60)
print("Ahora corriendo TODO el histórico (14.359 partidos)...")

# Reset y correr todo
ratings = {}
for _, row in df.iterrows():
    home, away = row["home_team"], row["away_team"]
    rh, ra, _, _, _ = elo_update(get(home), get(away), row["home_goals"], row["away_goals"])
    ratings[home] = rh
    ratings[away] = ra

print("\nTOP 15 equipos por rating FINAL (2025-05-25):")
top = sorted(ratings.items(), key=lambda kv: -kv[1])[:15]
for i, (t, r) in enumerate(top, 1):
    print(f"  {i:2d}. {t:25s} {r:7.1f}")

print("\nBOTTOM 5 (equipos peor rateados):")
bot = sorted(ratings.items(), key=lambda kv: kv[1])[:5]
for i, (t, r) in enumerate(bot, 1):
    print(f"  {i:2d}. {t:25s} {r:7.1f}")

# Guardar ratings finales
import json
with open("/home/Projects/futbol_elo_poisson/mini/mini_final_elo.json", "w") as f:
    json.dump({t: round(r, 2) for t, r in ratings.items()}, f, indent=2, ensure_ascii=False)
print(f"\n149 ratings finales guardados.")
