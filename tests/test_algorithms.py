from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tangle_sim.algorithms import MCMC, OperationalSBWRW, URTS
from tangle_sim.algorithms.base import SelectionContext, WalkStepLimitExceeded, sector_distance, stable_softmax
from tangle_sim.algorithms.sb_wrw import normative_sb_wrw_walk
from tangle_sim.domain import Tangle


def add(tangle: Tangle, parents: tuple[int, ...], time: float, sector: int = 0, branch: str | None = None) -> int:
    return tangle.add_transaction(
        issue_time=time, attachment_time=time, visible_time=time,
        issuer_id=sector, sector_id=sector, position=(sector / 10, 0.2),
        parents=parents, branch_id=branch,
    )


def context(sector: int = 0, grid: tuple[int, int] = (2, 3), time: float = 100.0) -> SelectionContext:
    return SelectionContext(0, sector, (0.05, 0.05), time, grid)


def selector(cls: type[MCMC] | type[OperationalSBWRW], alpha: float = 0.0, steps: int = 100) -> MCMC | OperationalSBWRW:
    return cls(alpha=alpha, entry_point_depth_seconds=100.0, max_walk_steps=steps, max_tip_retries=8)


def test_urts_is_approximately_uniform_without_replacement() -> None:
    tangle = Tangle()
    for sector in range(4):
        add(tangle, (0,), 1.0 + sector / 10, sector)
    view = tangle.view(10)
    rng = np.random.default_rng(2)
    urts = URTS()
    counts = {tip: 0 for tip in view.tips()}
    for _ in range(12000):
        counts[urts.select_tip(view, context(time=10), rng)] += 1
    expected = 12000 / 4
    chi_square = sum((value - expected) ** 2 / expected for value in counts.values())
    assert chi_square < 16.3  # relaxed 0.001 threshold, avoids a fragile exact-frequency test


def test_mcmc_alpha_zero_uniform_and_high_alpha_favors_weight() -> None:
    tangle = Tangle()
    low = add(tangle, (0,), 1.0, 0)
    high = add(tangle, (0,), 1.1, 0)
    add(tangle, (high,), 2.0, 0)
    view = tangle.view(10)
    zero = selector(MCMC, 0.0)
    assert np.allclose(zero.transition_probabilities(view, 0, (low, high)), (0.5, 0.5))
    biased = selector(MCMC, 10.0)
    probabilities = biased.transition_probabilities(view, 0, (low, high))
    assert probabilities[1] > 0.999


def test_stable_softmax_with_large_weights() -> None:
    probabilities = stable_softmax([1e300, 1e300 - 1e290, 1e300 - 2e290])
    assert np.all(np.isfinite(probabilities))
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[0] == pytest.approx(1.0)
    assert probabilities[0] >= probabilities[1] >= probabilities[2]


def test_same_sector_has_absolute_priority() -> None:
    tangle = Tangle()
    local = add(tangle, (0,), 1.0, 0)
    remote = add(tangle, (0,), 1.1, 5)
    for i in range(2, 12):
        remote = add(tangle, (remote,), float(i), 5)
    sb = selector(OperationalSBWRW, 100.0)
    eligible, ring = sb.eligible_candidates(tangle.view(20), context(0), (local, tangle.transactions[remote].parents[0]))
    assert eligible == (local,)
    assert ring == 0


def test_expands_to_nearest_nonempty_ring() -> None:
    tangle = Tangle()
    adjacent = add(tangle, (0,), 1.0, 1)
    far = add(tangle, (0,), 1.1, 5)
    eligible, ring = selector(OperationalSBWRW).eligible_candidates(tangle.view(5), context(0), (adjacent, far))
    assert eligible == (adjacent,)
    assert ring == 1


def test_wrw_weighting_and_alpha_zero_inside_ring() -> None:
    tangle = Tangle()
    first = add(tangle, (0,), 1.0, 1)
    second = add(tangle, (0,), 1.1, 3)  # both distance one from sector 0 in 2x3
    add(tangle, (second,), 2.0, 3)
    view = tangle.view(5)
    sb_zero = selector(OperationalSBWRW, 0.0)
    assert np.allclose(sb_zero.transition_probabilities(view, 0, (first, second)), (0.5, 0.5))
    sb = selector(OperationalSBWRW, np.log(3.0))
    assert sb.transition_probabilities(view, 0, (first, second))[1] == pytest.approx(0.75)


def test_sb_mode_divergence_counts_filter_exclusion_not_random_draw() -> None:
    tangle = Tangle()
    local_heavy = add(tangle, (0,), 1.0, 0)
    local_light = add(tangle, (0,), 1.1, 0)
    add(tangle, (local_heavy,), 2.0, 0)
    remote = add(tangle, (0,), 2.1, 1)
    add(tangle, (remote,), 3.0, 1)
    add(tangle, (remote,), 3.1, 1)
    view = tangle.view(10)
    sb = selector(OperationalSBWRW, 0.0)

    # For issuer sector zero, the global maximum is remote and is excluded by
    # local-first. The random draw among local candidates must not add another
    # kind of divergence.
    sb.walk(view, context(0, (1, 2), 10), np.random.default_rng(3))
    assert sb.diagnostics.mcmc_mode_divergence_steps == 1
    assert local_light in view.direct_approvers(0)


def test_observed_sb_frequency_matches_theoretical_transition() -> None:
    tangle = Tangle()
    low_tip = add(tangle, (0,), 1.0, 1)
    heavy_branch = add(tangle, (0,), 1.1, 3)
    heavy_tip = add(tangle, (heavy_branch,), 2.0, 3)
    view = tangle.view(5)
    sb = selector(OperationalSBWRW, np.log(3.0))
    rng = np.random.default_rng(918)
    heavy = sum(sb.walk(view, context(0, time=5), rng).endpoint == heavy_tip for _ in range(10000))
    assert heavy / 10000 == pytest.approx(0.75, abs=0.02)
    assert low_tip in view.tips()


def test_one_sector_matches_mcmc_distribution_pathwise() -> None:
    tangle = Tangle()
    a = add(tangle, (0,), 1.0)
    b = add(tangle, (0,), 1.1)
    add(tangle, (a,), 2.0)
    add(tangle, (b,), 2.1)
    view = tangle.view(10)
    mcmc = selector(MCMC, 0.3)
    sb = selector(OperationalSBWRW, 0.3)
    ctx = context(0, (1, 1), 10)
    for seed in range(500):
        assert mcmc.walk(view, ctx, np.random.default_rng(seed)).endpoint == sb.walk(view, ctx, np.random.default_rng(seed)).endpoint


def test_remote_only_path_does_not_stop_early() -> None:
    tangle = Tangle()
    remote = add(tangle, (0,), 1.0, 5)
    tip = add(tangle, (remote,), 2.0, 4)
    outcome = selector(OperationalSBWRW).walk(tangle.view(5), context(0, time=5), np.random.default_rng(1))
    assert outcome.endpoint == tip
    assert outcome.steps == 2


def test_sb_walk_respects_delayed_snapshot() -> None:
    tangle = Tangle()
    parent = add(tangle, (0,), 1.0, 0)
    tangle.add_transaction(
        issue_time=2.0, attachment_time=2.0, visible_time=10.0,
        issuer_id=0, sector_id=0, position=(0.1, 0.1), parents=(parent,),
    )
    outcome = selector(OperationalSBWRW).walk(tangle.view(5), context(0, time=5), np.random.default_rng(1))
    assert outcome.endpoint == parent


def test_second_tip_is_distinct_and_compatible() -> None:
    tangle = Tangle()
    add(tangle, (0,), 1.0, 0, "honest")
    add(tangle, (0,), 1.1, 1, "honest")
    parents = selector(MCMC).select_parents(tangle.view(5), context(time=5), np.random.default_rng(7))
    assert len(parents) == 2
    assert len(set(parents)) == 2
    assert {tangle.transactions[p].branch_id for p in parents} == {"honest"}


def test_optimized_matches_normative_reference() -> None:
    tangle = Tangle()
    layer = [0]
    for time in range(1, 8):
        next_layer = []
        for sector in range(3):
            parents = (layer[sector % len(layer)],)
            next_layer.append(add(tangle, parents, float(time), sector))
        layer = next_layer
    view = tangle.view(20)
    sb = selector(OperationalSBWRW, 0.2)
    ctx = context(1, (1, 3), 20)
    entry = sb.entry_point(view, ctx)
    for seed in range(1000):
        actual = sb.walk(view, ctx, np.random.default_rng(seed)).endpoint
        expected = normative_sb_wrw_walk(view, 1, entry, 0.2, np.random.default_rng(seed), (1, 3))
        assert actual == expected


def test_walk_step_limit_is_observable() -> None:
    tangle = Tangle()
    first = add(tangle, (0,), 1.0)
    add(tangle, (first,), 2.0)
    mcmc = selector(MCMC, steps=1)
    with pytest.raises(WalkStepLimitExceeded):
        mcmc.walk(tangle.view(5), context(time=5), np.random.default_rng(1))
    assert mcmc.diagnostics.step_limit_exceeded == 1


def test_sector_distance_is_manhattan() -> None:
    assert sector_distance(0, 5, (2, 3)) == 3
    assert sector_distance(3, 5, (2, 3)) == 2
