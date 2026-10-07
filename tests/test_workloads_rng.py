from __future__ import annotations

import numpy as np

from tangle_sim.config import load_config
from tangle_sim.simulation.engine import Simulator
from tangle_sim.simulation.rng import (
    LazyIndependentGenerators,
    SeedManager,
    independent_generators,
)
from tangle_sim.simulation.workloads import (
    _mmpp_times,
    generate_schedule,
    schedule_digest,
    workload_rate_intervals,
)


def test_common_random_numbers_same_exogenous_sequence() -> None:
    config = load_config("configs/article.yaml")
    simulator = Simulator(config)
    scenario = config.scenarios[1]
    first = simulator.common_inputs(scenario, 3, 30.0)
    second = simulator.common_inputs(scenario, 3, 30.0)
    assert first == second
    assert schedule_digest(first[1]) == schedule_digest(second[1])


def test_streams_are_not_accidentally_reused() -> None:
    seeds = SeedManager(123)
    identifiers = {
        seeds.identifier("PACTCare", 0, stream, "mcmc")
        for stream in ("workload", "topology", "delay", "selection", "attack", "measurement", "bootstrap")
    }
    assert len(identifiers) == 7
    a, b = independent_generators(np.random.default_rng(5), 2)
    assert not np.array_equal(a.integers(0, 2**32, 20), b.integers(0, 2**32, 20))


def test_lazy_children_match_eager_seed_sequence_children() -> None:
    eager_parent = np.random.default_rng(85)
    lazy_parent = np.random.default_rng(85)
    eager = independent_generators(eager_parent, 35)
    lazy = LazyIndependentGenerators.from_generator(lazy_parent)
    for index in (0, 1, 7, 34):
        assert np.array_equal(
            eager[index].integers(0, 2**32, 16),
            lazy[index].integers(0, 2**32, 16),
        )


def test_poisson_empirical_rate_within_statistical_tolerance() -> None:
    config = load_config("configs/article.yaml")
    scenario = config.scenarios[0]
    schedule = generate_schedule(scenario, 10000.0, np.random.default_rng(8), np.random.default_rng(9))
    empirical = len(schedule) / 10000.0
    assert abs(empirical - 4.0) < 0.08
    assert all(event.delay > 0 for event in schedule)


def test_all_workload_types_are_ordered_and_bounded() -> None:
    config = load_config("configs/article.yaml")
    for index, scenario in enumerate(config.scenarios):
        schedule = generate_schedule(scenario, 60.0, np.random.default_rng(index), np.random.default_rng(index + 20))
        times = [event.issue_time for event in schedule]
        assert times == sorted(times)
        assert all(0 < time <= 60 for time in times)


def test_schedule_prefix_is_invariant_to_horizon_for_every_workload() -> None:
    config = load_config("configs/article.yaml")
    for index, scenario in enumerate(config.scenarios):
        short = generate_schedule(
            scenario, 10.0,
            np.random.default_rng(100 + index), np.random.default_rng(200 + index),
        )
        long = generate_schedule(
            scenario, 40.0,
            np.random.default_rng(100 + index), np.random.default_rng(200 + index),
        )
        assert short == tuple(event for event in long if event.issue_time <= 10.0)


def test_tlip_mmpp_starts_in_stationary_distribution() -> None:
    config = load_config("configs/article.yaml")
    values = config.scenarios[1].values
    horizon = 1.0
    runs = 2000
    arrivals = sum(
        len(_mmpp_times(values, horizon, np.random.default_rng(seed)))
        for seed in range(runs)
    )
    empirical_rate = arrivals / (runs * horizon)
    # The stationary target is 0.75*8 + 0.25*24 = 12 tx/s.
    assert abs(empirical_rate - 12.0) < 0.5


def test_periodic_burst_rate_profile_tracks_quiet_and_burst_windows() -> None:
    config = load_config("configs/article.yaml")
    scenario = config.scenarios[2]
    simulator = Simulator(config)
    intervals = workload_rate_intervals(
        scenario,
        35.0,
        simulator.seeds.generator(scenario.name, 0, "workload"),
    )

    assert (0.0, 5.0, 140.0) in intervals
    assert (5.0, 30.0, 20.0) in intervals
    assert (30.0, 35.0, 140.0) in intervals
