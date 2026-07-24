"""Métricas de evaluación probabilística — proper scoring rules.

Extraído de fase1_baseline.py para reutilizar en Fase 3+.
"""
from __future__ import annotations

import numpy as np

RESULT_TO_IDX = {"H": 0, "D": 1, "A": 2}


def log_loss_1x2(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-15) -> float:
    """Log-loss estándar. probs: N×3 [P(H), P(D), P(A)]."""
    y_idx = np.array([RESULT_TO_IDX[r] for r in y_true])
    n = len(y_idx)
    p_true = np.clip(probs[np.arange(n), y_idx], eps, 1.0)
    return -np.log(p_true).mean()


def brier_1x2(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Brier score multi-clase. Suma de (p_i - y_i)^2 promediada."""
    y_idx = np.array([RESULT_TO_IDX[r] for r in y_true])
    n = len(y_idx)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y_idx] = 1.0
    return ((probs - onehot) ** 2).sum(axis=1).mean()


def rps_1x2(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Ranked Probability Score — castiga distancia entre clases ordenadas H-D-A."""
    y_idx = np.array([RESULT_TO_IDX[r] for r in y_true])
    n = len(y_idx)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y_idx] = 1.0
    cum_p = np.cumsum(probs[:, :2], axis=1)
    cum_y = np.cumsum(onehot[:, :2], axis=1)
    return ((cum_p - cum_y) ** 2).sum(axis=1).mean()


def all_metrics(y_true: np.ndarray, probs: np.ndarray, label: str = "") -> dict:
    return {
        "model": label,
        "n": len(y_true),
        "log_loss": log_loss_1x2(y_true, probs),
        "rps": rps_1x2(y_true, probs),
        "brier": brier_1x2(y_true, probs),
    }
