"""Unit tests for the Dixon-Coles bivariate Poisson module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from dixon_coles import (
    dc_tau,
    matrix_to_1x2,
    matrix_to_btts,
    matrix_to_over_under,
    score_matrix,
)


def test_dc_tau_identity_outside_low_cells():
    """For (i,j) not in {(0,0),(1,0),(0,1),(1,1)}, tau must be 1."""
    for i in range(2, 5):
        for j in range(0, 5):
            assert dc_tau(i, j, 1.5, 1.2, rho=-0.05) == 1.0


def test_score_matrix_sums_to_one():
    """The matrix must be a valid probability distribution."""
    m = score_matrix(1.5, 1.2, rho=-0.05, max_goals=8)
    assert abs(m.sum() - 1.0) < 1e-9


def test_1x2_sums_to_one():
    """P(H) + P(D) + P(A) must equal 1."""
    m = score_matrix(1.5, 1.2, rho=-0.05)
    ph, pd, pa = matrix_to_1x2(m)
    assert abs(ph + pd + pa - 1.0) < 1e-9


def test_1x2_home_advantage_higher_lambda_wins():
    """λ_home > λ_away → P(H) > P(A)."""
    m = score_matrix(1.8, 1.0, rho=-0.05)
    ph, pd, pa = matrix_to_1x2(m)
    assert ph > pa


def test_over_under_complementary():
    """P(over) + P(under) must equal 1."""
    m = score_matrix(1.5, 1.2, rho=-0.05)
    o, u = matrix_to_over_under(m, threshold=2.5)
    assert abs(o + u - 1.0) < 1e-9


def test_btts_in_bounds():
    m = score_matrix(1.5, 1.2, rho=-0.05)
    p = matrix_to_btts(m)
    assert 0.0 <= p <= 1.0


def test_dc_rho_zero_recovers_independent_poisson():
    """With rho=0, matrix must equal outer product of Poissons (approx)."""
    from scipy.stats import poisson

    lh, la = 1.5, 1.2
    m = score_matrix(lh, la, rho=0.0, max_goals=8)
    expected = np.outer(poisson.pmf(range(9), lh), poisson.pmf(range(9), la))
    expected /= expected.sum()  # normalize to match our matrix's normalization
    np.testing.assert_allclose(m, expected, atol=1e-6)
