from __future__ import annotations

from collections import Counter
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from math import hypot
from typing import Protocol

import numpy as np

from tangle_sim.domain import InconsistentViewError, TangleView
from tangle_sim.simulation.rng import LazyIndependentGenerators


class WalkStepLimitExceeded(RuntimeError):
    pass


class ExcludedEndpointReached(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SelectionContext:
    issuer_id: int
    issuer_sector: int
    issuer_position: tuple[float, float]
    snapshot_time: float
    grid: tuple[int, int]
    required_branch: str | None = None
    entry_point_override: int | None = None
    required_ancestor: int | None = None


@dataclass(frozen=True, slots=True)
class WalkOutcome:
    endpoint: int
    steps: int


@dataclass
class SelectorDiagnostics:
    selections: int = 0
    walks: int = 0
    walk_lengths: list[int] = field(default_factory=list)
    second_tip_retries: int = 0
    fallbacks: int = 0
    single_parent: int = 0
    compatibility_restrictions: int = 0
    ring_steps: Counter[int] = field(default_factory=Counter)
    sector_distance_sum: float = 0.0
    sector_distance_max: int = 0
    endpoint_euclidean_distances: list[float] = field(default_factory=list)
    endpoint_sectors: Counter[int] = field(default_factory=Counter)
    mcmc_mode_divergence_steps: int = 0
    sb_steps: int = 0
    step_limit_exceeded: int = 0

    def record_walk(self, outcome: WalkOutcome) -> None:
        self.walks += 1
        self.walk_lengths.append(outcome.steps)

    def as_dict(self) -> dict[str, float | int | str]:
        lengths = np.asarray(self.walk_lengths, dtype=float)
        total_rings = sum(self.ring_steps.values())
        values: dict[str, float | int | str] = {
            "tsa_selections": self.selections,
            "walks": self.walks,
            "walk_length_mean": float(lengths.mean()) if lengths.size else 0.0,
            "walk_length_median": float(np.median(lengths)) if lengths.size else 0.0,
            "walk_length_p95": float(np.quantile(lengths, 0.95)) if lengths.size else 0.0,
            "walk_length_max": int(lengths.max()) if lengths.size else 0,
            "second_tip_retries": self.second_tip_retries,
            "fallbacks": self.fallbacks,
            "single_parent_selections": self.single_parent,
            "compatibility_restrictions": self.compatibility_restrictions,
            "sector_step_distance_mean": self.sector_distance_sum / total_rings if total_rings else 0.0,
            "sector_step_distance_max": self.sector_distance_max,
            "endpoint_euclidean_mean": float(np.mean(self.endpoint_euclidean_distances)) if self.endpoint_euclidean_distances else 0.0,
            "endpoint_sector_distribution": ";".join(f"{k}:{v}" for k, v in sorted(self.endpoint_sectors.items())),
            "sb_mcmc_mode_divergence_percent": 100.0 * self.mcmc_mode_divergence_steps / self.sb_steps if self.sb_steps else 0.0,
            "walk_step_limit_exceeded": self.step_limit_exceeded,
        }
        values.update({
            "ring_r0_count": self.ring_steps[0],
            "ring_r1_count": self.ring_steps[1],
            "ring_r2_count": self.ring_steps[2],
            "ring_r3plus_count": sum(v for k, v in self.ring_steps.items() if k >= 3),
        })
        for label, count in (
            ("r0", self.ring_steps[0]),
            ("r1", self.ring_steps[1]),
            ("r2", self.ring_steps[2]),
            ("r3plus", sum(v for k, v in self.ring_steps.items() if k >= 3)),
        ):
            values[f"ring_{label}_percent"] = 100.0 * count / total_rings if total_rings else 0.0
        return values


class TipSelector(Protocol):
    name: str
    diagnostics: SelectorDiagnostics

    def select_tip(self, view: TangleView, context: SelectionContext, rng: np.random.Generator, excluded_tips: frozenset[int] = frozenset()) -> int: ...
    def select_parents(self, view: TangleView, context: SelectionContext, rng: np.random.Generator) -> tuple[int, ...]: ...


def stable_softmax(logits: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("softmax requires a nonempty finite one-dimensional vector")
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("invalid stabilized softmax normalization")
    return weights / total


def wrw_transition_probabilities(
    view: TangleView,
    current: int,
    candidates: tuple[int, ...] | list[int],
    alpha: float,
) -> np.ndarray:
    """Normative WRW transition shared by evolution and exact measurement.

    Writing the full expression here makes the sign auditable.  The reference
    weight is constant across candidates, so this is numerically equivalent to
    ``softmax(alpha * CW(candidate))`` while directly matching the adopted
    literature formula ``exp[-alpha * (CW(current) - CW(candidate))]``.
    """
    if not candidates:
        raise ValueError("WRW transition requires at least one candidate")
    reference_weight = view.cumulative_weight(current)
    logits = [
        -float(alpha) * (reference_weight - view.cumulative_weight(candidate))
        for candidate in candidates
    ]
    return stable_softmax(logits)


def sector_distance(a: int, b: int, grid: tuple[int, int]) -> int:
    rows, cols = grid
    if not 0 <= a < rows * cols or not 0 <= b < rows * cols:
        raise ValueError("sector outside configured grid")
    ar, ac = divmod(a, cols)
    br, bc = divmod(b, cols)
    return abs(ar - br) + abs(ac - bc)


class RandomWalkSelector:
    name = "base"

    def __init__(self, *, alpha: float, entry_point_depth_seconds: float, max_walk_steps: int, max_tip_retries: int, entry_point_policy: str = "genesis") -> None:
        if alpha < 0 or entry_point_depth_seconds < 0 or max_walk_steps < 1 or max_tip_retries < 0:
            raise ValueError("invalid walk selector parameter")
        if entry_point_policy not in {"genesis", "time_depth"}:
            raise ValueError("entry_point_policy must be genesis or time_depth")
        self.alpha = float(alpha)
        self.entry_point_depth_seconds = float(entry_point_depth_seconds)
        self.max_walk_steps = int(max_walk_steps)
        self.max_tip_retries = int(max_tip_retries)
        self.entry_point_policy = entry_point_policy
        self.diagnostics = SelectorDiagnostics()

    def entry_point(self, view: TangleView, context: SelectionContext) -> int:
        if context.entry_point_override is not None:
            entry = int(context.entry_point_override)
            if not view.is_visible(entry) or not view.is_valid(entry):
                raise InconsistentViewError(f"attack entry point {entry} is not visible and valid")
            return entry
        if self.entry_point_policy == "genesis":
            return 0
        threshold = context.snapshot_time - self.entry_point_depth_seconds
        index = bisect_right(view.tangle.attachment_times, threshold) - 1
        while index > 0:
            if view.is_visible(index) and view.is_valid(index) and view.is_branch_compatible(index, context.required_branch):
                return index
            index -= 1
        return 0

    def walk(self, view: TangleView, context: SelectionContext, rng: np.random.Generator, excluded_tips: frozenset[int] = frozenset()) -> WalkOutcome:
        raise NotImplementedError

    def select_tip(self, view: TangleView, context: SelectionContext, rng: np.random.Generator, excluded_tips: frozenset[int] = frozenset()) -> int:
        outcome = self.walk(view, context, rng, excluded_tips)
        self.diagnostics.record_walk(outcome)
        return outcome.endpoint

    def _endpoint_fallback(self, view: TangleView, context: SelectionContext, rng: np.random.Generator, excluded: frozenset[int]) -> int | None:
        tips = [tip for tip in view.tips(context.required_branch) if tip not in excluded]
        if not tips:
            return None
        probs = stable_softmax([self.alpha * view.cumulative_weight(tip) for tip in tips])
        return int(rng.choice(tips, p=probs))

    def select_parents(self, view: TangleView, context: SelectionContext, rng: np.random.Generator) -> tuple[int, ...]:
        self.diagnostics.selections += 1
        streams = LazyIndependentGenerators.from_generator(rng)
        first = self.select_tip(view, context, streams[0])
        branch = view.branch(first)
        second_context = replace(context, required_branch=branch)
        if branch is not None:
            self.diagnostics.compatibility_restrictions += 1
        second: int | None = None
        for retry in range(self.max_tip_retries + 1):
            try:
                candidate = self.select_tip(view, second_context, streams[retry + 1], frozenset({first}))
            except ExcludedEndpointReached:
                self.diagnostics.second_tip_retries += 1
                continue
            if candidate != first:
                second = candidate
                break
            self.diagnostics.second_tip_retries += 1
        if second is None:
            self.diagnostics.fallbacks += 1
            second = self._endpoint_fallback(
                view,
                second_context,
                streams[self.max_tip_retries + 2],
                frozenset({first}),
            )
        parents = (first,) if second is None else (first, second)
        if len(parents) == 1:
            self.diagnostics.single_parent += 1
        for endpoint in parents:
            tx = view.tangle.transactions[endpoint]
            self.diagnostics.endpoint_sectors[tx.sector_id] += 1
            self.diagnostics.endpoint_euclidean_distances.append(
                hypot(context.issuer_position[0] - tx.position[0], context.issuer_position[1] - tx.position[1])
            )
        return parents


def terminal_outcome(
    view: TangleView,
    x: int,
    steps: int,
    excluded_tips: frozenset[int],
    required_branch: str | None = None,
) -> WalkOutcome:
    if x in excluded_tips:
        raise ExcludedEndpointReached(x)
    if not view.is_tip(x, required_branch):
        raise InconsistentViewError(f"transaction {x} has no eligible approvers but is not a view tip")
    return WalkOutcome(x, steps)
