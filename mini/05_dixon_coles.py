"""
MINI paso 5 — DIXON-COLES corrección aplicada a Poisson bivariable.

Vemos la matriz 9x9 P(goles_home=i, goles_away=j) ANTES y DESPUÉS de la corrección.
"""
import numpy as np
from scipy.stats import poisson

def dc_tau(i, j, lh, la, rho):
    if i == 0 and j == 0: return 1.0 - lh * la * rho
    if i == 0 and j == 1: return 1.0 + lh * rho
    if i == 1 and j == 0: return 1.0 + la * rho
    if i == 1 and j == 1: return 1.0 - rho
    return 1.0

def matriz(lh, la, rho, size=9):
    p_h = poisson.pmf(np.arange(size), lh)
    p_a = poisson.pmf(np.arange(size), la)
    m = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            m[i, j] = p_h[i] * p_a[j] * dc_tau(i, j, lh, la, rho)
    return m / m.sum()

# Ejemplo: partido moderado con home ligeramente favorito
lh, la, rho = 1.5, 1.0, -0.03

print("=" * 70)
print("  MINI paso 5 — DIXON-COLES (partido λ_home=1.5, λ_away=1.0)")
print("=" * 70)

mat_puro = matriz(lh, la, rho=0.0)
mat_dc = matriz(lh, la, rho=rho)

def print_mat(m, title):
    print(f"\n{title}")
    header = "  ".join(f"A={j}" for j in range(6))
    print(f"        {header}")
    for i in range(6):
        row = "  ".join(f"{m[i,j]:.3f}" for j in range(6))
        print(f"  H={i} {row}")

print_mat(mat_puro, "Poisson INDEPENDIENTE (rho=0, sin DC):")
print_mat(mat_dc, "Con corrección DIXON-COLES (rho=-0.03):")

# Diferencias en las 4 celdas ajustadas
print(f"\n─── Cambios por celda (solo las 4 celdas ajustadas) ───")
for (i, j) in [(0,0), (1,0), (0,1), (1,1)]:
    d = mat_dc[i,j] - mat_puro[i,j]
    pct = (d / mat_puro[i,j]) * 100
    print(f"  P({i}-{j}):  {mat_puro[i,j]:.4f} → {mat_dc[i,j]:.4f}  (Δ = {d:+.4f}, {pct:+.1f}%)")

# Mercados derivados
def mercados(m):
    n = m.shape[0]
    ph = sum(m[i,j] for i in range(n) for j in range(n) if i>j)
    pd_ = sum(m[i,i] for i in range(n))
    pa = sum(m[i,j] for i in range(n) for j in range(n) if i<j)
    over = sum(m[i,j] for i in range(n) for j in range(n) if i+j > 2)
    btts = sum(m[i,j] for i in range(1,n) for j in range(1,n))
    return ph, pd_, pa, over, btts

ph1, pd1, pa1, o1, b1 = mercados(mat_puro)
ph2, pd2, pa2, o2, b2 = mercados(mat_dc)

print(f"\n─── Comparación mercados ───")
print(f"  Mercado                Poisson puro    Con DC")
print(f"  P(Home gana)             {ph1:.3f}          {ph2:.3f}")
print(f"  P(Empate)                {pd1:.3f}          {pd2:.3f}  ← DC lo sube (0-0 y 1-1 ↑)")
print(f"  P(Away gana)             {pa1:.3f}          {pa2:.3f}")
print(f"  P(Over 2.5)              {o1:.3f}          {o2:.3f}")
print(f"  P(Ambos marcan)          {b1:.3f}          {b2:.3f}")

print("\n─── Interpretación ───")
print("  DC empuja probabilidad hacia los resultados bajos (0-0, 1-0, 0-1, 1-1)")
print("  que la literatura fut. muestra están infra-modelados por Poisson puro.")
print("  Por eso P(empate) sube ligeramente cuando aplicamos DC.")
