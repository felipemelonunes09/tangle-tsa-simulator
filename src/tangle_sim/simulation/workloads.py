from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from tangle_sim.config import ScenarioConfig
from .rng import independent_generators


@dataclass(frozen=True, slots=True)
class Issuer:
    id: int
    sector_id: int
    position: tuple[float, float]


@dataclass(frozen=True, slots=True)
class ExogenousTransaction:
    issue_time: float
    issuer_id: int
    delay: float


def generate_issuers(scenario: ScenarioConfig, rng: np.random.Generator) -> tuple[Issuer, ...]:
    rows, cols = scenario.grid
    sectors = rows * cols
    mode = scenario.values.get("sector_distribution", "balanced")
    if mode == "balanced":
        assignments = np.arange(scenario.issuers) % sectors
        rng.shuffle(assignments)
    elif mode == "heterogeneous":
        weights = rng.dirichlet(np.linspace(0.8, 2.0, sectors))
        assignments = rng.choice(sectors, size=scenario.issuers, p=weights)
    elif mode == "clustered":
        weights = rng.dirichlet(np.full(sectors, 0.65))
        assignments = rng.choice(sectors, size=scenario.issuers, p=weights)
    else:
        raise ValueError(f"unknown sector distribution: {mode}")
    issuers: list[Issuer] = []
    for issuer_id, sector_value in enumerate(assignments):
        sector = int(sector_value)
        row, col = divmod(sector, cols)
        if mode == "clustered":
            x = np.clip((col + 0.5 + rng.normal(0.0, 0.14)) / cols, col / cols, (col + 1) / cols)
            y = np.clip((row + 0.5 + rng.normal(0.0, 0.14)) / rows, row / rows, (row + 1) / rows)
        else:
            x = (col + rng.random()) / cols
            y = (row + rng.random()) / rows
        issuers.append(Issuer(issuer_id, sector, (float(x), float(y))))
    return tuple(issuers)


def _poisson_times(rate: float, horizon: float, rng: np.random.Generator, start: float = 0.0) -> list[float]:
    if rate <= 0:
        return []
    result: list[float] = []
    t = float(start)
    while True:
        t += float(rng.exponential(1.0 / rate))
        if t > horizon:
            break
        result.append(t)
    return result


def _mmpp_times_and_rate_intervals(
    values: dict[str, object],
    horizon: float,
    rng: np.random.Generator,
) -> tuple[list[float], tuple[tuple[float, float, float], ...]]:
    rates = (float(values["low_rate"]), float(values["high_rate"]))
    means = (float(values["low_mean_duration"]), float(values["high_mean_duration"]))
    # Start the two-state CTMC in equilibrium. Always starting in the low state
    # creates a severe transient in short diagnostic runs (45 s mean low-state
    # duration in TLIP), despite the configured stationary mean being 12 tx/s.
    low_probability = means[0] / sum(means)
    state = 0 if rng.random() < low_probability else 1
    t = 0.0
    state_end = float(rng.exponential(means[state]))
    result: list[float] = []
    intervals: list[tuple[float, float, float]] = []
    interval_start = 0.0
    while t < horizon:
        arrival = t + float(rng.exponential(1.0 / rates[state]))
        if arrival < state_end:
            t = arrival
            if t <= horizon:
                result.append(t)
        else:
            if state_end > interval_start:
                intervals.append((
                    interval_start,
                    min(state_end, horizon),
                    rates[state],
                ))
            t = state_end
            interval_start = state_end
            state = 1 - state
            state_end = t + float(rng.exponential(means[state]))
    if interval_start < horizon:
        intervals.append((interval_start, horizon, rates[state]))
    return result, tuple(
        interval for interval in intervals if interval[1] > interval[0]
    )


def _mmpp_times(values: dict[str, object], horizon: float, rng: np.random.Generator) -> list[float]:
    times, _ = _mmpp_times_and_rate_intervals(values, horizon, rng)
    return times


def _periodic_burst_times(values: dict[str, object], horizon: float, rng: np.random.Generator) -> list[float]:
    background_rng, burst_seed_rng = independent_generators(rng, 2)
    background = _poisson_times(float(values["background_rate"]), horizon, background_rng)
    period = float(values["burst_period"])
    duration = float(values["burst_seconds"])
    burst_rate = float(values["burst_rate"])
    jitter = float(values.get("jitter_seconds", 0.0))
    result = list(background)
    start = 0.0
    while start < horizon:
        # Each burst window owns independent arrival and jitter streams. Generate
        # the complete last window and filter afterward, so extending the
        # observation horizon never changes an already generated prefix.
        arrival_rng, jitter_rng = independent_generators(burst_seed_rng, 2)
        end = start + duration
        lower = max(start, float(np.nextafter(0.0, 1.0)))
        for t in _poisson_times(burst_rate, end, arrival_rng, start=start):
            shifted = float(np.clip(t + jitter_rng.normal(0.0, jitter), lower, end))
            if shifted < horizon:
                result.append(shifted)
        start += period
    result.sort()
    return result


def generate_schedule(
    scenario: ScenarioConfig,
    horizon: float,
    workload_rng: np.random.Generator,
    delay_rng: np.random.Generator,
) -> tuple[ExogenousTransaction, ...]:
    # Isolate arrival times from issuer choices. Without this split, requesting
    # a longer horizon consumes extra arrival draws before issuer assignment and
    # retroactively changes the supposedly common prefix for the same seed.
    time_rng, issuer_rng = independent_generators(workload_rng, 2)
    if scenario.workload == "poisson":
        times = _poisson_times(float(scenario.values["rate"]), horizon, time_rng)
    elif scenario.workload == "mmpp":
        times = _mmpp_times(scenario.values, horizon, time_rng)
    elif scenario.workload == "periodic_burst":
        times = _periodic_burst_times(scenario.values, horizon, time_rng)
    else:
        raise ValueError(f"unknown workload: {scenario.workload}")
    result: list[ExogenousTransaction] = []
    for issue_time in times:
        issuer = int(issuer_rng.integers(0, scenario.issuers))
        # The lognormal is strictly positive and parameterized by its median.
        delay = float(delay_rng.lognormal(math.log(scenario.delay_median), scenario.delay_sigma))
        result.append(ExogenousTransaction(float(issue_time), issuer, delay))
    return tuple(result)


def workload_rate_intervals(
    scenario: ScenarioConfig,
    horizon: float,
    workload_rng: np.random.Generator,
) -> tuple[tuple[float, float, float], ...]:
    """Return the honest workload intensity profile for an attack replica.

    The MMPP state path is replayed from the same immutable workload seed as
    the honest schedule. Periodic bursts use their configured windows. This
    lets an adversarial issuance share track contemporaneous honest load rather
    than a long-run average that is wrong during a low state or between bursts.
    """

    if horizon <= 0:
        return ()
    time_rng, _ = independent_generators(workload_rng, 2)
    if scenario.workload == "poisson":
        return ((0.0, horizon, float(scenario.values["rate"])),)
    if scenario.workload == "mmpp":
        _, intervals = _mmpp_times_and_rate_intervals(
            scenario.values, horizon, time_rng,
        )
        return intervals
    if scenario.workload == "periodic_burst":
        background_rate = float(scenario.values["background_rate"])
        burst_rate = float(scenario.values["burst_rate"])
        burst_seconds = float(scenario.values["burst_seconds"])
        burst_period = float(scenario.values["burst_period"])
        boundaries = {0.0, horizon}
        start = 0.0
        while start < horizon:
            boundaries.add(start)
            boundaries.add(min(start + burst_seconds, horizon))
            start += burst_period
        ordered = sorted(boundaries)
        intervals: list[tuple[float, float, float]] = []
        for left, right in zip(ordered, ordered[1:]):
            in_burst = left % burst_period < burst_seconds - 1e-12
            intervals.append((
                left,
                right,
                background_rate + (burst_rate if in_burst else 0.0),
            ))
        return tuple(intervals)
    raise ValueError(f"unknown workload: {scenario.workload}")


def schedule_digest(schedule: tuple[ExogenousTransaction, ...]) -> str:
    h = hashlib.sha256()
    for event in schedule:
        h.update(f"{event.issue_time:.17g},{event.issuer_id},{event.delay:.17g}\n".encode("ascii"))
    return h.hexdigest()
