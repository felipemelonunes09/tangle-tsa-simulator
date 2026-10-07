from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
from typing import Any

import numpy as np

from tangle_sim.algorithms import MCMC, OperationalSBWRW, URTS
from tangle_sim.algorithms.base import SelectionContext, TipSelector
from tangle_sim.config import ExperimentConfig, ScenarioConfig, scenario_alpha
from tangle_sim.domain import Tangle
from .events import Event, EventKind, PendingAttachment
from .rng import SeedManager
from .workloads import ExogenousTransaction, Issuer, generate_issuers, generate_schedule, schedule_digest


@dataclass(slots=True)
class SimulationResult:
    tangle: Tangle
    final_time: float
    topology: tuple[Issuer, ...]
    schedule: tuple[ExogenousTransaction, ...]
    diagnostics: dict[str, Any]


def make_selector(name: str, config: ExperimentConfig, scenario: ScenarioConfig) -> TipSelector:
    mcmc = config.raw["algorithms"]["mcmc"]
    alpha = scenario_alpha(config, scenario.name)
    common = dict(
        alpha=alpha,
        entry_point_policy=str(mcmc["entry_point_policy"]),
        entry_point_depth_seconds=float(mcmc["entry_point_depth_seconds"]),
        max_walk_steps=int(mcmc["max_walk_steps"]),
        max_tip_retries=int(mcmc["max_tip_retries"]),
    )
    if name == "urts":
        return URTS()
    if name == "mcmc":
        return MCMC(**common)
    if name == "sb_wrw":
        sb = config.raw["algorithms"]["sb_wrw"]
        if sb["conformance"] != "operational_reproduction":
            raise ValueError("paper_exact implementation is unavailable without full-text traceability")
        return OperationalSBWRW(
            alpha=alpha,
            entry_point_policy=common["entry_point_policy"],
            entry_point_depth_seconds=common["entry_point_depth_seconds"],
            max_walk_steps=int(sb["max_walk_steps"]),
            max_tip_retries=int(sb["max_tip_retries"]),
        )
    raise ValueError(f"unknown selector: {name}")


class Simulator:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.seeds = SeedManager(config.root_seed)

    def common_inputs(self, scenario: ScenarioConfig, replicate: int, horizon: float) -> tuple[tuple[Issuer, ...], tuple[ExogenousTransaction, ...]]:
        topology = generate_issuers(scenario, self.seeds.generator(scenario.name, replicate, "topology"))
        schedule = generate_schedule(
            scenario,
            horizon,
            self.seeds.generator(scenario.name, replicate, "workload"),
            self.seeds.generator(scenario.name, replicate, "delay"),
        )
        return topology, schedule

    def run_benign(
        self,
        scenario: ScenarioConfig,
        algorithm: str,
        replicate: int,
        topology: tuple[Issuer, ...] | None = None,
        schedule: tuple[ExogenousTransaction, ...] | None = None,
    ) -> SimulationResult:
        sim = self.config.raw["simulation"]
        warmup_end = float(sim["warmup_seconds"])
        measurement_end = warmup_end + float(sim["measurement_seconds"])
        final_time = measurement_end + float(sim["drain_seconds"])
        if topology is None or schedule is None:
            # The preregistered drain is quiescent: stop issuing at the end of
            # measurement, but finish attachment events already in flight.
            topology, schedule = self.common_inputs(scenario, replicate, measurement_end)
        tangle = Tangle()
        selector = make_selector(algorithm, self.config, scenario)
        selection_rng = self.seeds.generator(scenario.name, replicate, "selection", algorithm)
        queue: list[Event] = []
        counter = itertools.count()
        for external in schedule:
            heapq.heappush(queue, Event(external.issue_time, next(counter), EventKind.ISSUE, external))

        while queue:
            event = heapq.heappop(queue)
            if event.time > final_time:
                break
            if event.kind == EventKind.ISSUE:
                external = event.payload
                issuer = topology[external.issuer_id]
                view = tangle.view(event.time)
                context = SelectionContext(
                    issuer_id=issuer.id,
                    issuer_sector=issuer.sector_id,
                    issuer_position=issuer.position,
                    snapshot_time=event.time,
                    grid=scenario.grid,
                )
                parents = selector.select_parents(view, context, selection_rng)
                pending = PendingAttachment(event.time, issuer.id, parents)
                heapq.heappush(queue, Event(event.time + external.delay, next(counter), EventKind.ATTACH, pending))
            elif event.kind == EventKind.ATTACH:
                pending = event.payload
                issuer = topology[pending.issuer_id]
                tangle.add_transaction(
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
        tangle.assert_invariants()
        diagnostics = selector.diagnostics.as_dict()
        diagnostics.update({
            "external_schedule_sha256": schedule_digest(schedule),
            "external_transactions": len(schedule),
            "warmup_window_transactions": sum(
                external.issue_time < warmup_end for external in schedule
            ),
            "measurement_window_transactions": sum(
                warmup_end <= external.issue_time < measurement_end for external in schedule
            ),
            "drain_window_transactions": sum(
                measurement_end <= external.issue_time <= final_time for external in schedule
            ),
            "attachments_completed_during_drain": sum(
                tx.id != 0
                and tx.issue_time < measurement_end
                and tx.attachment_time >= measurement_end
                for tx in tangle.transactions
            ),
            "attached_transactions": len(tangle) - 1,
        })
        return SimulationResult(tangle, final_time, topology, schedule, diagnostics)
