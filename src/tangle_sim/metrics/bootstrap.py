from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt

import numpy as np


@dataclass(frozen=True, slots=True)
class Interval:
    estimate: float
    low: float
    high: float


def bootstrap_mean(values: np.ndarray | list[float], rng: np.random.Generator, samples: int = 10000, confidence: float = 0.95) -> Interval:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0 or np.any(~np.isfinite(x)):
        raise ValueError("bootstrap requires finite observations")
    indices = rng.integers(0, x.size, size=(samples, x.size))
    estimates = x[indices].mean(axis=1)
    q = (1.0 - confidence) / 2.0
    return Interval(float(x.mean()), float(np.quantile(estimates, q)), float(np.quantile(estimates, 1 - q)))


def paired_bootstrap_difference(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, samples: int, confidence: float = 0.95) -> Interval:
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired arrays must be one-dimensional and equal length")
    return bootstrap_mean(a - b, rng, samples, confidence)


def paired_effect_size(a: np.ndarray, b: np.ndarray) -> float:
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = differences.std(ddof=1) if len(differences) > 1 else 0.0
    return float(differences.mean() / sd) if sd > 0 else 0.0


def paired_normal_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) < 2:
        return 1.0
    se = d.std(ddof=1) / sqrt(len(d))
    if se == 0:
        return 0.0 if d.mean() != 0 else 1.0
    z = abs(float(d.mean() / se))
    return float(erfc(z / sqrt(2.0)))


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted

