from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import heapq
import itertools

import numpy as np

from tangle_sim.algorithms.base import (
    SelectionContext,
    sector_distance,
    wrw_transition_probabilities,
)
from tangle_sim.config import (
    ExperimentConfig,
    ScenarioConfig,
    scenario_alpha,
    scenario_private_build_seconds,
)
from tangle_sim.domain import Tangle
from .engine import SimulationResult, Simulator, make_selector
from .events import Event, EventKind, PendingAttachment
from .rng import independent_generators
from .workloads import (
    ExogenousTransaction,
    Issuer,
    schedule_digest,
    workload_rate_intervals,
)


@dataclass(frozen=True, slots=True)
class AttackMetrics:
    event_observed: bool
    time_to_event_or_censoring: float
    censoring_horizon: float
    fraud_success: int
    fraud_confidence_final: float
    honest_confidence_final: float
    fraud_tx_id: int
    honest_tx_id: int

    def as_dict(self) -> dict[str, bool | int | float]:
        return {
            "event_observed": self.event_observed,
            "time_to_event_or_censoring": self.time_to_event_or_censoring,
            "censoring_horizon": self.censoring_horizon,
            "fraud_success": self.fraud_success,
            "fraud_confidence_final": self.fraud_confidence_final,
            "honest_confidence_final": self.honest_confidence_final,
            "fraud_tx_id": self.fraud_tx_id,
            "honest_tx_id": self.honest_tx_id,
        }


@dataclass(slots=True)
class AttackResult:
    simulation: SimulationResult
    metrics: AttackMetrics
    confidence_history: tuple[tuple[float, float, float], ...]


def _nominal_rate(scenario: ScenarioConfig) -> float:
    v = scenario.values
    if scenario.workload == "poisson":
        return float(v["rate"])
    if scenario.workload == "mmpp":
        low_duration = float(v["low_mean_duration"])
        high_duration = float(v["high_mean_duration"])
        return (
            float(v["low_rate"]) * low_duration + float(v["high_rate"]) * high_duration
        ) / (low_duration + high_duration)
    return float(v["background_rate"]) + float(v["burst_rate"]) * float(v["burst_seconds"]) / float(v["burst_period"])


def _attack_times(start: float, end: float, rate: float, rng: np.random.Generator) -> list[float]:
    if rate <= 0:
        return []
    values: list[float] = []
    t = start
    while True:
        t += float(rng.exponential(1.0 / rate))
        if t > end:
            return values
        values.append(t)


def _attacker_schedule(
    start: float,
    end: float,
    rate: float,
    topology: tuple[Issuer, ...],
    rng: np.random.Generator,
    issuer_policy: str,
    rate_intervals: tuple[tuple[float, float, float], ...] | None = None,
    attachment_delay_policy: str = "instant",
    delay_median: float = 0.0,
    delay_sigma: float = 0.0,
) -> tuple[int, tuple[tuple[float, int], ...]]:
    """Generate a common adversarial schedule, including its spatial origin.

    ``topology_proportional`` models a distributed coalition/Sybil adversary:
    every malicious issue is assigned to an issuer location drawn uniformly
    from the scenario topology.  Time and issuer draws use independent child
    streams so extending the observation horizon preserves the earlier prefix.
    ``single_fixed`` retains the historical single-device model and its exact
    time stream for backward compatibility.
    """

    def issue_times(time_rng: np.random.Generator) -> list[float]:
        if rate_intervals is None:
            return _attack_times(start, end, rate, time_rng)
        values: list[float] = []
        for interval_start, interval_end, interval_rate in rate_intervals:
            left = max(start, interval_start)
            right = min(end, interval_end)
            if right > left:
                values.extend(
                    _attack_times(left, right, interval_rate, time_rng)
                )
        values.sort()
        return values

    def delayed(
        schedule: tuple[tuple[float, int], ...],
        delay_rng: np.random.Generator,
    ) -> tuple[tuple[float, int], ...]:
        if attachment_delay_policy == "instant":
            return schedule
        shifted = tuple(sorted(
            (
                time + float(
                    delay_rng.lognormal(
                        np.log(delay_median), delay_sigma,
                    )
                ),
                issuer_id,
            )
            for time, issuer_id in schedule
        ))
        return tuple(item for item in shifted if item[0] <= end)

    if issuer_policy == "single_fixed":
        issuer_id = topology[-1].id
        if attachment_delay_policy == "instant":
            times = issue_times(rng)
            return issuer_id, tuple((time, issuer_id) for time in times)
        time_rng, delay_rng = independent_generators(rng, 2)
        times = issue_times(time_rng)
        schedule = tuple((time, issuer_id) for time in times)
        return issuer_id, delayed(schedule, delay_rng)

    child_count = 2 if attachment_delay_policy == "instant" else 3
    children = independent_generators(rng, child_count)
    time_rng, issuer_rng = children[:2]
    times = issue_times(time_rng)
    root_issuer_id = int(issuer_rng.integers(0, len(topology)))
    schedule = tuple(
        (time, int(issuer_rng.integers(0, len(topology))))
        for time in times
    )
    if attachment_delay_policy == "instant":
        return root_issuer_id, schedule
    return root_issuer_id, delayed(schedule, children[2])


def _attacker_schedule_digest(
    root_issuer_id: int,
    schedule: tuple[tuple[float, int], ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"root,{root_issuer_id}\n".encode("ascii"))
    for issue_time, issuer_id in schedule:
        digest.update(f"{issue_time:.17g},{issuer_id}\n".encode("ascii"))
    return digest.hexdigest()


def _branch_confidence(
    tangle: Tangle,
    tx_id: int,
    honest_tx_id: int,
    at_time: float,
    algorithm: str,
    config: ExperimentConfig,
    scenario: ScenarioConfig,
    topology: tuple[Issuer, ...],
    rng: np.random.Generator,
    cache: dict[str, object] | None = None,
) -> tuple[float, float]:
    del rng
    view = tangle.view(at_time)
    if algorithm == "urts":
        tips = view.tips()
        if not tips:
            return 0.0, 0.0
        scale = 1.0 / len(tips)
        fraud = sum(scale for tip in tips if view.approves(tip, tx_id))
        honest = sum(scale for tip in tips if view.approves(tip, honest_tx_id))
        return float(fraud), float(honest)
    if algorithm not in {"mcmc", "sb_wrw"}:
        raise ValueError(f"unknown algorithm: {algorithm}")

    # Once a walk enters either conflict branch, branch inheritance guarantees
    # that every possible endpoint remains in that same branch.  Solving the
    # absorbing DAG inside those already classified subgraphs is redundant.
    # Dynamic programming is therefore needed only on the neutral prefix and
    # any neutral side tips.  This is algebraically identical to propagating a
    # full endpoint vector, but makes exact event-driven checks tractable.
    fraud_bit = 1 << tx_id
    honest_bit = 1 << honest_tx_id
    cached_size = int(cache.get("classified_size", 0)) if cache is not None else 0
    if cache is not None and cached_size:
        neutral_ids = cache["neutral_ids"]
        assert isinstance(neutral_ids, list)
        for candidate in view.tangle.transactions[cached_size:]:
            if (
                view.is_visible(candidate.id)
                and not candidate.past_mask & fraud_bit
                and not candidate.past_mask & honest_bit
            ):
                neutral_ids.append(candidate.id)
        cache["classified_size"] = len(view.tangle)
    else:
        neutral_ids = [
            candidate.id
            for candidate in view.tangle.transactions
            if view.is_visible(candidate.id)
            and not candidate.past_mask & fraud_bit
            and not candidate.past_mask & honest_bit
        ]
        if cache is not None:
            cache["neutral_ids"] = neutral_ids
            cache["classified_size"] = len(view.tangle)
    neutral_set = set(neutral_ids)
    alpha = scenario_alpha(config, scenario.name)
    eligibility_cache: dict[tuple[int, int], tuple[tuple[int, ...], tuple[int, ...]]]
    if cache is None:
        eligibility_cache = {}
    else:
        eligibility_cache = cache.setdefault("sb_eligibility", {})
        assert isinstance(eligibility_cache, dict)

    def classified_probability(
        candidate: int,
        probabilities: dict[int, float],
        branch_bit: int,
    ) -> float:
        if candidate in neutral_set:
            return probabilities[candidate]
        return float(
            bool(view.tangle.transactions[candidate].past_mask & branch_bit)
        )

    def solve(issuer_sector: int) -> tuple[float, float]:
        fraud_probability: dict[int, float] = {}
        honest_probability: dict[int, float] = {}
        for current in reversed(neutral_ids):
            candidates = view.direct_approvers(current)
            if not candidates:
                fraud_probability[current] = 0.0
                honest_probability[current] = 0.0
                continue
            eligible = candidates
            if algorithm == "sb_wrw":
                cache_key = (current, issuer_sector)
                cached = eligibility_cache.get(cache_key)
                if cached is not None and cached[0] == candidates:
                    eligible = cached[1]
                else:
                    nearest = min(
                        sector_distance(
                            issuer_sector,
                            view.sector(candidate),
                            scenario.grid,
                        )
                        for candidate in candidates
                    )
                    eligible = tuple(
                        candidate
                        for candidate in candidates
                        if sector_distance(
                            issuer_sector,
                            view.sector(candidate),
                            scenario.grid,
                        )
                        == nearest
                    )
                    eligibility_cache[cache_key] = (candidates, eligible)
            if len(eligible) == 1:
                only = eligible[0]
                fraud_probability[current] = classified_probability(
                    only, fraud_probability, fraud_bit,
                )
                honest_probability[current] = classified_probability(
                    only, honest_probability, honest_bit,
                )
                continue
            probabilities = wrw_transition_probabilities(
                view, current, eligible, alpha,
            )
            fraud_probability[current] = sum(
                float(probability)
                * classified_probability(
                    candidate, fraud_probability, fraud_bit,
                )
                for candidate, probability in zip(
                    eligible, probabilities, strict=True,
                )
            )
            honest_probability[current] = sum(
                float(probability)
                * classified_probability(
                    candidate, honest_probability, honest_bit,
                )
                for candidate, probability in zip(
                    eligible, probabilities, strict=True,
                )
            )
        return fraud_probability.get(0, 0.0), honest_probability.get(0, 0.0)

    if algorithm == "mcmc":
        return solve(0)
    if cache is not None and "sector_counts" in cache:
        sector_counts = cache["sector_counts"]
        assert isinstance(sector_counts, Counter)
    else:
        sector_counts = Counter(issuer.sector_id for issuer in topology)
        if cache is not None:
            cache["sector_counts"] = sector_counts
    fraud = 0.0
    honest = 0.0
    for issuer_sector, count in sector_counts.items():
        sector_fraud, sector_honest = solve(issuer_sector)
        share = count / len(topology)
        fraud += share * sector_fraud
        honest += share * sector_honest
    return float(fraud), float(honest)


def run_attack(
    simulator: Simulator,
    scenario: ScenarioConfig,
    algorithm: str,
    replicate: int,
    topology: tuple[Issuer, ...] | None = None,
    schedule: tuple[ExogenousTransaction, ...] | None = None,
) -> AttackResult:
    config = simulator.config
    attack = config.raw["attack"]
    attack_start = float(config.raw["simulation"]["warmup_seconds"])
    private_build_seconds = scenario_private_build_seconds(
        config, scenario.name,
    )
    release_time = attack_start + private_build_seconds
    horizon = float(attack["attack_observation_seconds"])
    final_time = release_time + horizon
    if topology is None or schedule is None:
        topology, schedule = simulator.common_inputs(scenario, replicate, final_time)

    selection_rng = simulator.seeds.generator(scenario.name, replicate, "selection", algorithm)
    measurement_rng = simulator.seeds.generator(scenario.name, replicate, "measurement", algorithm)
    attack_rng = simulator.seeds.generator(scenario.name, replicate, "attack")
    attacker_fraction = float(attack["attacker_rate_fraction"])
    fraction_schedule = tuple(
        (
            float(step["after_seconds"]),
            float(step["fraction"]),
        )
        for step in attack.get("attacker_rate_fraction_schedule", ())
    )
    rate_semantics = str(attack.get("attacker_rate_semantics", "honest_rate_ratio"))
    rate_profile = str(
        attack.get("attacker_rate_profile", "nominal_average")
    )
    nominal_attacker_rate = (
        attacker_fraction / (1.0 - attacker_fraction) * _nominal_rate(scenario)
        if rate_semantics == "total_issuance_share"
        else attacker_fraction * _nominal_rate(scenario)
    )
    continue_after_release = bool(attack.get("continue_attacker_after_release", True))
    stop_after_consolidation = bool(
        attack.get("stop_after_fraud_consolidation", False)
    )
    parasite_strategy = str(attack.get("parasite_chain_strategy", "linear_legacy"))
    if parasite_strategy == "tsa_adaptive":
        effective_parasite_strategy = (
            "urts_tip_flood"
            if algorithm == "urts"
            else "simple_multi_pinned"
        )
    elif parasite_strategy == "tsa_adaptive_v2":
        effective_parasite_strategy = {
            "urts": "urts_tip_flood",
            "mcmc": "simple_1_pinned",
            "sb_wrw": "sector_hybrid_multi_pinned",
        }[algorithm]
    else:
        effective_parasite_strategy = parasite_strategy
    parasite_root_policy = str(
        attack.get("parasite_root_policy", "visible_tips")
    )
    sector_merge_interval = int(
        attack.get("sector_merge_interval_by_scenario", {}).get(
            scenario.name, attack.get("sector_merge_interval", 2),
        )
    )
    detection_mode = str(attack.get("fraud_detection_mode", "periodic"))
    attacker_issuer_policy = str(
        attack.get("attacker_issuer_policy", "single_fixed")
    )
    attachment_delay_policy = str(
        attack.get("attacker_attachment_delay_policy", "instant")
    )
    conflict_issuer_policy = str(
        attack.get("conflict_issuer_policy", "legacy_split")
    )
    attacker_end = final_time if continue_after_release else release_time
    attacker_rate_intervals: tuple[tuple[float, float, float], ...] | None = None
    if rate_profile == "workload_matched" or fraction_schedule:
        honest_intervals = (
            workload_rate_intervals(
                scenario,
                final_time,
                simulator.seeds.generator(
                    scenario.name, replicate, "workload",
                ),
            )
            if rate_profile == "workload_matched"
            else ((0.0, final_time, _nominal_rate(scenario)),)
        )
        escalation_boundaries = tuple(
            release_time + after_seconds
            for after_seconds, _ in fraction_schedule
        )

        def fraction_at(at_time: float) -> float:
            fraction = attacker_fraction
            elapsed = at_time - release_time
            for after_seconds, scheduled_fraction in fraction_schedule:
                if elapsed + 1e-12 < after_seconds:
                    break
                fraction = scheduled_fraction
            return fraction

        profiled_intervals: list[tuple[float, float, float]] = []
        for left, right, honest_rate in honest_intervals:
            boundaries = [
                boundary
                for boundary in escalation_boundaries
                if left < boundary < right
            ]
            points = (left, *boundaries, right)
            for interval_left, interval_right in zip(points, points[1:]):
                interval_fraction = fraction_at(interval_left)
                multiplier = (
                    interval_fraction / (1.0 - interval_fraction)
                    if rate_semantics == "total_issuance_share"
                    else interval_fraction
                )
                profiled_intervals.append((
                    interval_left,
                    interval_right,
                    multiplier * honest_rate,
                ))
        attacker_rate_intervals = tuple(profiled_intervals)

    def expected_rate(left: float, right: float) -> float:
        if right <= left:
            return 0.0
        if attacker_rate_intervals is None:
            return nominal_attacker_rate
        expected_count = sum(
            max(0.0, min(right, interval_end) - max(left, interval_start))
            * interval_rate
            for interval_start, interval_end, interval_rate
            in attacker_rate_intervals
        )
        return expected_count / (right - left)

    attacker_rate = expected_rate(attack_start, attacker_end)
    attacker_root_issuer_id, attacker_schedule = _attacker_schedule(
        attack_start,
        attacker_end,
        attacker_rate,
        topology,
        attack_rng,
        attacker_issuer_policy,
        attacker_rate_intervals,
        attachment_delay_policy,
        scenario.delay_median,
        scenario.delay_sigma,
    )
    attacker_times = tuple(time for time, _ in attacker_schedule)

    tangle = Tangle()
    selector = make_selector(algorithm, config, scenario)
    queue: list[Event] = []
    counter = itertools.count()
    for external in schedule:
        heapq.heappush(queue, Event(external.issue_time, next(counter), EventKind.ISSUE, external))
    heapq.heappush(queue, Event(attack_start, next(counter), EventKind.ATTACK_START))
    for time, issuer_id in attacker_schedule:
        heapq.heappush(
            queue,
            Event(time, next(counter), EventKind.ATTACK_ISSUE, issuer_id),
        )
    check_interval = float(attack["confidence_check_interval_seconds"])
    if detection_mode == "periodic":
        check_time = release_time
        while check_time <= final_time + 1e-12:
            heapq.heappush(
                queue,
                Event(check_time, next(counter), EventKind.CONFIDENCE_CHECK),
            )
            check_time += check_interval
    else:
        # Publication changes visibility without inserting a transaction, and
        # the administrative horizon needs an explicit final observation.
        heapq.heappush(
            queue,
            Event(release_time, next(counter), EventKind.CONFIDENCE_CHECK),
        )
        if final_time > release_time:
            heapq.heappush(
                queue,
                Event(final_time, next(counter), EventKind.CONFIDENCE_CHECK),
            )

    honest_tx_id: int | None = None
    fraud_tx_id: int | None = None
    last_fraud_id: int | None = None
    last_fraud_by_sector: dict[int, int] = {}
    malicious_issue_count = 0
    parasite_root_id: int | None = None
    parasite_roots_by_sector: dict[int, int] = {}
    high_since: float | None = None
    consolidated_at: float | None = None
    consolidated_crossing_at: float | None = None
    max_qualifying_duration = 0.0
    history: list[tuple[float, float, float]] = []
    honest_weight_at_release: float | None = None
    fraud_weight_at_release: float | None = None
    confidence_evaluations = 0
    branch_confidence_cache: dict[str, object] = {}
    fraud_zero_absorbing_shortcut = False

    def evaluate_confidence(at_time: float) -> None:
        nonlocal high_since, consolidated_at, consolidated_crossing_at
        nonlocal max_qualifying_duration, honest_weight_at_release
        nonlocal fraud_weight_at_release, confidence_evaluations
        nonlocal fraud_zero_absorbing_shortcut
        if fraud_tx_id is None or honest_tx_id is None:
            return
        fraud_conf, honest_conf = _branch_confidence(
            tangle,
            fraud_tx_id,
            honest_tx_id,
            at_time,
            algorithm,
            config,
            scenario,
            topology,
            measurement_rng,
            branch_confidence_cache,
        )
        confidence_evaluations += 1
        if (
            algorithm == "sb_wrw"
            and not continue_after_release
            and abs(at_time - release_time) <= 1e-12
            and fraud_conf == 0.0
        ):
            # With no future adversarial emission, zero SB-WRW probability is
            # absorbing: every populated issuer sector excludes every route to
            # F0. New non-fraud approvers cannot create such a route because
            # direct approvers are never removed and the nearest-ring distance
            # can only stay equal or decrease. A runtime guard below disables
            # the shortcut if a fraud descendant nevertheless appears.
            fraud_zero_absorbing_shortcut = True
        observation = (at_time - release_time, fraud_conf, honest_conf)
        if history and abs(history[-1][0] - observation[0]) <= 1e-12:
            history[-1] = observation
        else:
            history.append(observation)
        if honest_weight_at_release is None and at_time >= release_time:
            release_view = tangle.view(release_time)
            honest_weight_at_release = release_view.cumulative_weight(honest_tx_id)
            fraud_weight_at_release = release_view.cumulative_weight(fraud_tx_id)

        threshold = float(attack["fraud_confidence_threshold"])
        require_honest_rejection = bool(
            attack.get("require_honest_confidence_below_complement", True)
        )
        qualifying = fraud_conf >= threshold and (
            not require_honest_rejection
            or honest_conf <= (1.0 - threshold + 1e-12)
        )
        if not qualifying:
            high_since = None
            return
        if high_since is None:
            high_since = at_time
            if detection_mode == "event_driven":
                deadline = at_time + float(attack["confidence_stability_seconds"])
                if deadline <= final_time + 1e-12:
                    heapq.heappush(
                        queue,
                        Event(
                            deadline,
                            next(counter),
                            EventKind.CONFIDENCE_STABILITY_DEADLINE,
                            at_time,
                        ),
                    )
        qualifying_duration = at_time - high_since
        max_qualifying_duration = max(
            max_qualifying_duration,
            qualifying_duration,
        )
        if (
            consolidated_at is None
            and qualifying_duration
            >= float(attack["confidence_stability_seconds"]) - 1e-12
        ):
            consolidated_at = at_time
            consolidated_crossing_at = high_since

    processed_until = 0.0
    while queue:
        event = heapq.heappop(queue)
        if event.time > final_time:
            break
        processed_until = event.time
        if event.kind == EventKind.ISSUE:
            external = event.payload
            issuer = topology[external.issuer_id]
            before_release = honest_tx_id is not None and event.time < release_time
            context = SelectionContext(
                issuer.id,
                issuer.sector_id,
                issuer.position,
                event.time,
                scenario.grid,
                required_branch="honest" if before_release else None,
                entry_point_override=honest_tx_id if before_release else None,
                required_ancestor=honest_tx_id if before_release else None,
            )
            parents = selector.select_parents(tangle.view(event.time), context, selection_rng)
            heapq.heappush(
                queue,
                Event(event.time + external.delay, next(counter), EventKind.ATTACH, PendingAttachment(event.time, issuer.id, parents)),
            )
        elif event.kind == EventKind.ATTACH:
            pending = event.payload
            issuer = topology[pending.issuer_id]
            attached_tx_id = tangle.add_transaction(
                issue_time=pending.issue_time,
                attachment_time=event.time,
                visible_time=event.time if pending.visible_time is None else pending.visible_time,
                issuer_id=issuer.id,
                sector_id=issuer.sector_id,
                position=issuer.position,
                parents=pending.parents,
                honest=pending.honest,
                conflict_id=pending.conflict_id,
                branch_id=pending.branch_id,
            )
            if detection_mode == "event_driven" and event.time >= release_time:
                if (
                    fraud_zero_absorbing_shortcut
                    and fraud_tx_id is not None
                    and tangle.transactions[attached_tx_id].past_mask
                    & (1 << fraud_tx_id)
                ):
                    fraud_zero_absorbing_shortcut = False
                if not fraud_zero_absorbing_shortcut:
                    evaluate_confidence(event.time)
        elif event.kind == EventKind.ATTACK_START:
            attack_view = tangle.view(event.time)
            visible_tips = attack_view.tips()
            if effective_parasite_strategy in {
                "simple_multi_pinned",
                "sector_parallel_multi_pinned",
                "sector_hybrid_multi_pinned",
                "urts_tip_flood",
            }:
                if parasite_root_policy == "recent_visible_transactions":
                    root_candidates = tuple(
                        tx_id for tx_id in attack_view.visible_ids()
                        if tx_id != 0
                    ) or visible_tips or (0,)
                else:
                    root_candidates = visible_tips or (0,)
                occupied_sectors = sorted({issuer.sector_id for issuer in topology})
                for sector_id in occupied_sectors:
                    parasite_roots_by_sector[sector_id] = min(
                        root_candidates,
                        key=lambda tx_id: (
                            sector_distance(
                                sector_id,
                                tangle.transactions[tx_id].sector_id,
                                scenario.grid,
                            ),
                            -tangle.transactions[tx_id].attachment_time,
                            -tangle.transactions[tx_id].cumulative_weight,
                            -tx_id,
                        ),
                    )
                root_sector = topology[attacker_root_issuer_id].sector_id
                common_parent = parasite_roots_by_sector[root_sector]
            else:
                common_parent = int(visible_tips[0]) if visible_tips else 0
                parasite_roots_by_sector = {
                    issuer.sector_id: common_parent for issuer in topology
                }
            parasite_root_id = common_parent
            fraud_issuer = topology[attacker_root_issuer_id]
            honest_issuer = (
                fraud_issuer
                if conflict_issuer_policy == "same_attacker"
                else topology[0]
            )
            honest_tx_id = tangle.add_transaction(
                issue_time=event.time,
                attachment_time=event.time,
                visible_time=event.time,
                issuer_id=honest_issuer.id,
                sector_id=honest_issuer.sector_id,
                position=honest_issuer.position,
                parents=(common_parent,),
                honest=True,
                conflict_id="double_spend_1",
                branch_id="honest",
            )
            fraud_tx_id = tangle.add_transaction(
                issue_time=event.time,
                attachment_time=event.time,
                visible_time=release_time,
                issuer_id=fraud_issuer.id,
                sector_id=fraud_issuer.sector_id,
                position=fraud_issuer.position,
                parents=(common_parent,),
                honest=False,
                conflict_id="double_spend_1",
                branch_id="fraud",
            )
            last_fraud_id = fraud_tx_id
        elif event.kind == EventKind.ATTACK_ISSUE:
            malicious_issue_count += 1
            if last_fraud_id is None or parasite_root_id is None:
                continue
            fraud_issuer = topology[int(event.payload)]
            visible_time = release_time if event.time < release_time else event.time
            # Penzkofer et al., Sec. 3 / Fig. 6a: in a simple 1-pinned
            # parasite chain, every malicious transaction approves both the
            # previous PC transaction and the honest root to which the PC is
            # pinned.  The former implementation used only the previous PC
            # transaction and therefore modeled an ordinary linear side chain.
            if effective_parasite_strategy == "urts_tip_flood":
                pin_root = parasite_roots_by_sector[fraud_issuer.sector_id]
                if fraud_tx_id is None:
                    raise RuntimeError("tip-flood attack started before F0")
                parents = (fraud_tx_id, pin_root)
            elif effective_parasite_strategy == "simple_multi_pinned":
                pin_root = parasite_roots_by_sector[fraud_issuer.sector_id]
                parents = (last_fraud_id, pin_root)
            elif effective_parasite_strategy == "sector_parallel_multi_pinned":
                pin_root = parasite_roots_by_sector[fraud_issuer.sector_id]
                if fraud_tx_id is None:
                    raise RuntimeError("sector-parallel PC started before F0")
                previous = last_fraud_by_sector.get(
                    fraud_issuer.sector_id, fraud_tx_id,
                )
                parents = (previous, pin_root)
            elif effective_parasite_strategy == "sector_hybrid_multi_pinned":
                pin_root = parasite_roots_by_sector[fraud_issuer.sector_id]
                if fraud_tx_id is None:
                    raise RuntimeError("sector-hybrid PC started before F0")
                previous = last_fraud_by_sector.get(
                    fraud_issuer.sector_id, fraud_tx_id,
                )
                if (
                    malicious_issue_count % sector_merge_interval == 0
                    and last_fraud_id != previous
                ):
                    parents = (previous, last_fraud_id)
                else:
                    parents = (previous, pin_root)
            elif effective_parasite_strategy == "simple_1_pinned":
                parents = (last_fraud_id, parasite_root_id)
            else:
                parents = (last_fraud_id,)
            new_fraud_id = tangle.add_transaction(
                issue_time=event.time,
                attachment_time=event.time,
                visible_time=visible_time,
                issuer_id=fraud_issuer.id,
                sector_id=fraud_issuer.sector_id,
                position=fraud_issuer.position,
                parents=parents,
                honest=False,
                conflict_id="double_spend_1",
                branch_id="fraud",
            )
            last_fraud_id = new_fraud_id
            if effective_parasite_strategy in {
                "sector_parallel_multi_pinned",
                "sector_hybrid_multi_pinned",
            }:
                last_fraud_by_sector[fraud_issuer.sector_id] = new_fraud_id
            if detection_mode == "event_driven" and event.time >= release_time:
                evaluate_confidence(event.time)
        elif event.kind == EventKind.CONFIDENCE_CHECK:
            if (
                not fraud_zero_absorbing_shortcut
                or abs(event.time - final_time) <= 1e-12
            ):
                evaluate_confidence(event.time)
        elif event.kind == EventKind.CONFIDENCE_STABILITY_DEADLINE:
            # Ignore a stale deadline if an intervening DAG mutation broke the
            # qualifying interval or started a newer one.
            if high_since is not None and abs(high_since - float(event.payload)) <= 1e-12:
                evaluate_confidence(event.time)
        if stop_after_consolidation and consolidated_at is not None:
            break

    if fraud_tx_id is None or honest_tx_id is None:
        raise RuntimeError("attack initialization event was not processed")
    tangle.assert_invariants()
    final_fraud = history[-1][1] if history else 0.0
    final_honest = history[-1][2] if history else 0.0
    observed = consolidated_at is not None
    time_value = consolidated_at - release_time if observed else horizon
    metrics = AttackMetrics(
        event_observed=observed,
        time_to_event_or_censoring=float(time_value),
        censoring_horizon=horizon,
        fraud_success=int(observed),
        fraud_confidence_final=float(final_fraud),
        honest_confidence_final=float(final_honest),
        fraud_tx_id=fraud_tx_id,
        honest_tx_id=honest_tx_id,
    )
    diagnostics = selector.diagnostics.as_dict()
    diagnostics.update({
        "external_schedule_sha256": schedule_digest(schedule),
        "external_transactions": len(schedule),
        "attached_transactions": len(tangle) - 1,
        "attacker_transactions": len(attacker_times) + 1,
        "private_attacker_transactions": 1 + sum(
            time < release_time for time in attacker_times
        ),
        "post_release_attacker_transactions": sum(
            time >= release_time for time in attacker_times
        ),
        "attacker_transactions_processed": sum(
            not tx.honest for tx in tangle.transactions
        ),
        "post_release_attacker_transactions_processed": sum(
            not tx.honest
            and tx.id != fraud_tx_id
            and tx.attachment_time >= release_time
            for tx in tangle.transactions
        ),
        "attacker_continues_after_release": continue_after_release,
        "stop_after_fraud_consolidation": stop_after_consolidation,
        "attack_processed_until": processed_until,
        "attacker_rate_semantics": rate_semantics,
        "attacker_nominal_rate": attacker_rate,
        "attacker_rate_profile": rate_profile,
        "attacker_rate_fraction_schedule": ";".join(
            f"{after_seconds:g}:{fraction:g}"
            for after_seconds, fraction in fraction_schedule
        ),
        "attacker_rate_fraction_final": (
            fraction_schedule[-1][1]
            if fraction_schedule else attacker_fraction
        ),
        "private_expected_attacker_rate": expected_rate(
            attack_start, release_time,
        ),
        "post_release_expected_attacker_rate": expected_rate(
            release_time, attacker_end,
        ),
        "attacker_issuer_policy": attacker_issuer_policy,
        "attacker_attachment_delay_policy": attachment_delay_policy,
        "conflict_issuer_policy": conflict_issuer_policy,
        "conflict_same_issuer": (
            tangle.transactions[honest_tx_id].issuer_id
            == tangle.transactions[fraud_tx_id].issuer_id
        ),
        "honest_conflict_sector": tangle.transactions[honest_tx_id].sector_id,
        "fraud_conflict_sector": tangle.transactions[fraud_tx_id].sector_id,
        "attacker_schedule_sha256": _attacker_schedule_digest(
            attacker_root_issuer_id, attacker_schedule,
        ),
        "attacker_distinct_issuers": len({
            attacker_root_issuer_id,
            *(issuer_id for _, issuer_id in attacker_schedule),
        }),
        "attacker_distinct_sectors": len({
            topology[issuer_id].sector_id
            for issuer_id in {
                attacker_root_issuer_id,
                *(value for _, value in attacker_schedule),
            }
        }),
        "parasite_chain_strategy": parasite_strategy,
        "effective_parasite_chain_strategy": effective_parasite_strategy,
        "sector_merge_interval": sector_merge_interval,
        "private_build_seconds": private_build_seconds,
        "parasite_root_policy": parasite_root_policy,
        "parasite_root_tx_id": int(parasite_root_id),
        "parasite_root_count": len(set(parasite_roots_by_sector.values())),
        "parasite_links_to_roots": sum(
            1
            for tx in tangle.transactions
            if not tx.honest
            and tx.id != fraud_tx_id
            and len(tx.parents) == 2
            and tx.parents[1] in set(parasite_roots_by_sector.values())
        ),
        "parasite_links_to_root": (
            sum(
                1
                for tx_id in tangle.transactions[parasite_root_id].direct_approvers
                if not tangle.transactions[tx_id].honest
                and tx_id != fraud_tx_id
            )
            if parasite_root_id is not None else 0
        ),
        "attack_release_time": release_time,
        "honest_cumulative_weight_at_release": float(honest_weight_at_release or 0.0),
        "fraud_cumulative_weight_at_release": float(fraud_weight_at_release or 0.0),
        "fraud_to_honest_weight_ratio_at_release": (
            float(fraud_weight_at_release or 0.0) / float(honest_weight_at_release)
            if honest_weight_at_release else 0.0
        ),
        "fraud_consolidation_requires_honest_rejection": bool(
            attack.get("require_honest_confidence_below_complement", True)
        ),
        "fraud_detection_mode": detection_mode,
        "fraud_detection_resolution_seconds": (
            0.0 if detection_mode == "event_driven" else check_interval
        ),
        "fraud_confidence_evaluations": confidence_evaluations,
        "fraud_zero_absorbing_shortcut": fraud_zero_absorbing_shortcut,
        "fraud_event_time_definition": "end_of_stability_window_from_F0_publication",
        "fraud_threshold_crossing_time": (
            consolidated_crossing_at - release_time
            if consolidated_crossing_at is not None
            else None
        ),
        "fraud_confidence_max": max((item[1] for item in history), default=0.0),
        "fraud_confidence_max_time": next(
            (item[0] for item in history if item[1] == max((x[1] for x in history), default=0.0)),
            0.0,
        ),
        "fraud_qualifying_duration_max": max_qualifying_duration,
    })
    simulation = SimulationResult(
        tangle, processed_until, topology, schedule, diagnostics,
    )
    return AttackResult(simulation, metrics, tuple(history))
