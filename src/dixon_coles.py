"""
Dixon-Coles (1997): corrección de dependencia bajo-anotación sobre Poisson bivariable.

Corrige las 4 celdas (0,0), (1,0), (0,1), (1,1) que Poisson-independiente infra-modela.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson


def dc_tau(i: int, j: int, lam_home: float, lam_away: float, rho: float) -> float:
    """Función de corrección Dixon-Coles."""
    if i == 0 and j == 0:
        return 1.0 - lam_home * lam_away * rho
    if i == 0 and j == 1:
        return 1.0 + lam_home * rho
    if i == 1 and j == 0:
        return 1.0 + lam_away * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(lam_home: float, lam_away: float, rho: float = -0.05, max_goals: int = 8) -> np.ndarray:
    """Matriz P(home=i, away=j) con corrección DC. Filas=home, cols=away."""
    n = max_goals + 1
    m = np.zeros((n, n))
    p_home = poisson.pmf(np.arange(n), lam_home)
    p_away = poisson.pmf(np.arange(n), lam_away)
    for i in range(n):
        for j in range(n):
            m[i, j] = p_home[i] * p_away[j] * dc_tau(i, j, lam_home, lam_away, rho)
    # Renormalizar (DC hace la matriz no sumar exactamente 1)
    m /= m.sum()
    return m


def matrix_to_1x2(m: np.ndarray) -> tuple[float, float, float]:
    """P(H), P(D), P(A) desde matriz score."""
    n = m.shape[0]
    p_h = sum(m[i, j] for i in range(n) for j in range(n) if i > j)
    p_d = sum(m[i, i] for i in range(n))
    p_a = sum(m[i, j] for i in range(n) for j in range(n) if i < j)
    return p_h, p_d, p_a


def matrix_to_over_under(m: np.ndarray, threshold: float = 2.5) -> tuple[float, float]:
    """P(over_threshold), P(under_threshold)."""
    n = m.shape[0]
    p_over = sum(m[i, j] for i in range(n) for j in range(n) if (i + j) > threshold)
    return p_over, 1 - p_over


def matrix_to_btts(m: np.ndarray) -> float:
    """P(ambos marcan)."""
    n = m.shape[0]
    return sum(m[i, j] for i in range(1, n) for j in range(1, n))


if __name__ == "__main__":
    # Ejemplo: partido con λ_home=1.5, λ_away=1.0
    mat = score_matrix(1.5, 1.0, rho=-0.05)
    ph, pd, pa = matrix_to_1x2(mat)
    print(f"λ_home=1.5, λ_away=1.0, rho=-0.05:")
    print(f"  P(H)={ph:.3f}  P(D)={pd:.3f}  P(A)={pa:.3f}  suma={ph+pd+pa:.3f}")
    over, under = matrix_to_over_under(mat, 2.5)
    print(f"  P(over 2.5)={over:.3f}  P(under 2.5)={under:.3f}")
    print(f"  P(BTTS)={matrix_to_btts(mat):.3f}")
    print(f"  P(1-0)={mat[1,0]:.3f}  P(2-1)={mat[2,1]:.3f}  P(0-0)={mat[0,0]:.3f}")
