from __future__ import annotations

from collections import Counter

import numpy as np

from tangle_sim.algorithms.base import sector_distance, wrw_transition_probabilities
from tangle_sim.domain import TangleView
from tangle_sim.simulation.workloads import Issuer


def endpoint_distribution(
    view: TangleView,
    algorithm: str,
    *,
    alpha: float = 0.0,
    grid: tuple[int, int] = (1, 1),
    issuer_sector: int = 0,
    entry_point: int = 0,
) -> np.ndarray:
    """Compute exact absorbing endpoint probabilities on a visible DAG.

    Integer transaction IDs are topological because attachments are processed in
    event-time order. Probability mass can therefore be propagated once from the
    configured entry point toward direct approvers.
    """
    size = len(view.tangle)
    endpoints = np.zeros(size, dtype=float)
    if algorithm == "urts":
        tips = view.tips()
        if tips:
            endpoints[list(tips)] = 1.0 / len(tips)
        return endpoints
    if algorithm not in {"mcmc", "sb_wrw"}:
        raise ValueError(f"unknown algorithm: {algorithm}")
    if not view.is_visible(entry_point):
        raise ValueError("entry point is not visible")
    mass = np.zeros(size, dtype=float)
    mass[entry_point] = 1.0
    for tx_id in range(entry_point, size):
        current_mass = mass[tx_id]
        if current_mass == 0.0:
            continue
        candidates = view.direct_approvers(tx_id)
        if not candidates:
            endpoints[tx_id] += current_mass
            continue
        eligible = candidates
        if algorithm == "sb_wrw":
            nearest = min(sector_distance(issuer_sector, view.sector(candidate), grid) for candidate in candidates)
            eligible = tuple(
                candidate for candidate in candidates
                if sector_distance(issuer_sector, view.sector(candidate), grid) == nearest
            )
        probabilities = wrw_transition_probabilities(view, tx_id, eligible, alpha)
        for candidate, probability in zip(eligible, probabilities, strict=True):
            mass[candidate] += current_mass * float(probability)
    total = endpoints.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("exact endpoint distribution has invalid total mass")
    endpoints /= total
    return endpoints


def mixed_endpoint_distribution(
    view: TangleView,
    algorithm: str,
    topology: tuple[Issuer, ...],
    *,
    alpha: float,
    grid: tuple[int, int],
    entry_point: int = 0,
) -> np.ndarray:
    if algorithm != "sb_wrw":
        return endpoint_distribution(
            view, algorithm, alpha=alpha, grid=grid,
            issuer_sector=0, entry_point=entry_point,
        )
    sector_counts = Counter(issuer.sector_id for issuer in topology)
    result = np.zeros(len(view.tangle), dtype=float)
    for sector, count in sector_counts.items():
        result += (count / len(topology)) * endpoint_distribution(
            view, algorithm, alpha=alpha, grid=grid,
            issuer_sector=sector, entry_point=entry_point,
        )
    result /= result.sum()
    return result


def ancestor_confidences(view: TangleView, endpoint_probabilities: np.ndarray) -> np.ndarray:
    confidence = np.zeros(len(view.tangle), dtype=float)
    for endpoint, probability in enumerate(endpoint_probabilities):
        if probability == 0.0:
            continue
        mask = view.tangle.transactions[endpoint].past_mask
        while mask:
            lsb = mask & -mask
            confidence[lsb.bit_length() - 1] += probability
            mask ^= lsb
    return np.clip(confidence, 0.0, 1.0)
