from __future__ import annotations

import numpy as np
import pytest

from tangle_sim.metrics.bootstrap import bootstrap_mean, holm_adjust
from tangle_sim.metrics.survival import kaplan_meier, rmst


def test_kaplan_meier_manual_example_and_rmst() -> None:
    points = kaplan_meier([1.0, 2.0, 3.0], [True, False, True])
    assert points[1].survival == pytest.approx(2 / 3)
    assert points[2].time == 3.0
    assert points[2].survival == 0.0
    assert rmst([1.0, 2.0, 3.0], [True, False, True], 3.0) == pytest.approx(7 / 3)


def test_all_censored_rmst_equals_horizon() -> None:
    assert rmst([10, 10, 10], [False, False, False], 10) == 10


def test_all_immediate_events_have_zero_rmst() -> None:
    assert rmst([0, 0, 0], [True, True, True], 10) == 0


def test_bootstrap_is_deterministic() -> None:
    a = bootstrap_mean([1, 2, 3, 4], np.random.default_rng(55), 1000)
    b = bootstrap_mean([1, 2, 3, 4], np.random.default_rng(55), 1000)
    assert a == b


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
