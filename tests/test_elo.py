"""Unit tests for the recursive ELO module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from elo import EloState, expected_score, goal_diff_multiplier, update_elo


def test_expected_score_equal_ratings():
    """Equal ratings, no home advantage → expected 0.5."""
    assert abs(expected_score(1500, 1500, home_adv=0) - 0.5) < 1e-9


def test_expected_score_home_advantage():
    """Home advantage 65 points → expected > 0.5."""
    assert expected_score(1500, 1500, home_adv=65) > 0.5


def test_expected_score_stronger_team():
    """Higher-rated team → expected > 0.5."""
    assert expected_score(1600, 1500, home_adv=0) > 0.5
    assert expected_score(1500, 1600, home_adv=0) < 0.5


def test_goal_diff_multiplier_monotonic():
    """Multiplier must be non-decreasing with |goal difference|."""
    prev = 0
    for gd in range(0, 8):
        m = goal_diff_multiplier(gd)
        assert m >= prev
        prev = m


def test_update_elo_draw_equal_teams_no_change():
    """Equal-rating teams draw → no change (before home_adv)."""
    r_h, r_a = update_elo(1500, 1500, 1, 1, k=20, home_adv=0)
    assert abs(r_h - 1500) < 1e-9
    assert abs(r_a - 1500) < 1e-9


def test_update_elo_home_wins_gains_points():
    """Home team wins → gains, away loses same amount (zero-sum)."""
    r_h, r_a = update_elo(1500, 1500, 2, 0, k=20, home_adv=65)
    assert r_h > 1500
    assert r_a < 1500
    assert abs((r_h - 1500) + (r_a - 1500)) < 1e-9  # zero-sum


def test_elo_state_persists_across_matches():
    """State should carry ratings from match to match."""
    s = EloState(k=20, home_adv=65)
    s.update("A", "B", 1, 0)
    ra1 = s.get("A")
    rb1 = s.get("B")
    assert ra1 > 1500
    assert rb1 < 1500
    s.update("A", "C", 1, 0)
    ra2 = s.get("A")
    assert ra2 > ra1  # A got stronger
