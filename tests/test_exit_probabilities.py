from __future__ import annotations

import numpy as np
import pytest

from tangle_sim.algorithms import MCMC, OperationalSBWRW
from tangle_sim.algorithms.base import SelectionContext
from tangle_sim.domain import Tangle
from tangle_sim.metrics.exit_probabilities import ancestor_confidences, endpoint_distribution, mixed_endpoint_distribution
from tangle_sim.simulation.workloads import Issuer


def add(tangle: Tangle, parents: tuple[int, ...], time: float, sector: int = 0) -> int:
    return tangle.add_transaction(
        issue_time=time, attachment_time=time, visible_time=time,
        issuer_id=sector, sector_id=sector, position=(sector / 2, 0.25), parents=parents,
    )


def test_exact_mcmc_distribution_matches_theoretical_branch_probability() -> None:
    tangle = Tangle()
    low_tip = add(tangle, (0,), 1.0)
    heavy = add(tangle, (0,), 1.1)
    heavy_tip = add(tangle, (heavy,), 2.0)
    distribution = endpoint_distribution(tangle.view(5), "mcmc", alpha=np.log(3.0))
    assert distribution[low_tip] == pytest.approx(0.25)
    assert distribution[heavy_tip] == pytest.approx(0.75)
    assert distribution.sum() == pytest.approx(1.0)
    confidence = ancestor_confidences(tangle.view(5), distribution)
    assert confidence[0] == pytest.approx(1.0)
    assert confidence[heavy] == pytest.approx(0.75)


def test_exact_one_sector_sb_matches_mcmc() -> None:
    tangle = Tangle()
    left = add(tangle, (0,), 1.0)
    right = add(tangle, (0,), 1.1)
    add(tangle, (left,), 2.0)
    add(tangle, (right,), 2.1)
    view = tangle.view(5)
    mcmc = endpoint_distribution(view, "mcmc", alpha=0.2)
    sb = endpoint_distribution(view, "sb_wrw", alpha=0.2, grid=(1, 1), issuer_sector=0)
    assert np.allclose(mcmc, sb)


def test_sb_mixture_uses_actual_issuer_sector_distribution() -> None:
    tangle = Tangle()
    local_zero = add(tangle, (0,), 1.0, 0)
    local_one = add(tangle, (0,), 1.1, 1)
    topology = (
        Issuer(0, 0, (0.1, 0.1)),
        Issuer(1, 0, (0.2, 0.1)),
        Issuer(2, 0, (0.3, 0.1)),
        Issuer(3, 1, (0.8, 0.1)),
    )
    distribution = mixed_endpoint_distribution(
        tangle.view(5), "sb_wrw", topology,
        alpha=0.0, grid=(1, 2), entry_point=0,
    )
    assert distribution[local_zero] == pytest.approx(0.75)
    assert distribution[local_one] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("algorithm", "selector_type", "issuer_sector", "grid"),
    (
        ("mcmc", MCMC, 0, (1, 2)),
        ("sb_wrw", OperationalSBWRW, 0, (1, 2)),
    ),
)
def test_exact_distribution_matches_walk_sampling(
    algorithm: str,
    selector_type: type[MCMC] | type[OperationalSBWRW],
    issuer_sector: int,
    grid: tuple[int, int],
) -> None:
    tangle = Tangle()
    left = add(tangle, (0,), 1.0, 0)
    right = add(tangle, (0,), 1.1, 1)
    add(tangle, (left,), 2.0, 0)
    add(tangle, (right,), 2.1, 1)
    add(tangle, (right,), 2.2, 1)
    view = tangle.view(5)
    alpha = 0.3
    exact = endpoint_distribution(
        view, algorithm, alpha=alpha, grid=grid, issuer_sector=issuer_sector,
    )
    selector = selector_type(
        alpha=alpha, entry_point_policy="genesis", entry_point_depth_seconds=0.0,
        max_walk_steps=100, max_tip_retries=4,
    )
    context = SelectionContext(0, issuer_sector, (0.1, 0.1), 5.0, grid)
    rng = np.random.default_rng(9701)
    counts = np.zeros(len(tangle), dtype=float)
    for _ in range(20_000):
        counts[selector.walk(view, context, rng).endpoint] += 1
    observed = counts / counts.sum()
    assert np.allclose(observed, exact, atol=0.015)
