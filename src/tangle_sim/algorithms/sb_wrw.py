from __future__ import annotations

import numpy as np

from tangle_sim.domain import TangleView
from .base import (
    RandomWalkSelector,
    SelectionContext,
    WalkOutcome,
    WalkStepLimitExceeded,
    sector_distance,
    stable_softmax,
    terminal_outcome,
    wrw_transition_probabilities,
)


class OperationalSBWRW(RandomWalkSelector):
    """Normative operational reproduction of prompt section 5.3.

    This class must not be relabelled paper-exact without rule-by-rule full-text
    traceability. Distance filters; accumulated weight only weights candidates
    inside the nearest nonempty Manhattan ring.
    """

    name = "sb_wrw"
    conformance = "operational_reproduction"

    def eligible_candidates(self, view: TangleView, context: SelectionContext, candidates: tuple[int, ...] | list[int]) -> tuple[tuple[int, ...], int]:
        if not candidates:
            return (), 0
        nearest = context.grid[0] + context.grid[1]
        eligible: list[int] = []
        for candidate in candidates:
            distance = sector_distance(
                context.issuer_sector, view.sector(candidate), context.grid
            )
            if distance < nearest:
                nearest = distance
                eligible = [candidate]
            elif distance == nearest:
                eligible.append(candidate)
        return tuple(eligible), nearest

    def transition_probabilities(
        self,
        view: TangleView,
        current: int,
        eligible: tuple[int, ...] | list[int],
    ) -> np.ndarray:
        return wrw_transition_probabilities(view, current, eligible, self.alpha)

    def walk(self, view: TangleView, context: SelectionContext, rng: np.random.Generator, excluded_tips: frozenset[int] = frozenset()) -> WalkOutcome:
        x = self.entry_point(view, context)
        for step in range(self.max_walk_steps):
            candidates = view.direct_approvers(x, context.required_branch)
            if not candidates:
                return terminal_outcome(view, x, step, excluded_tips, context.required_branch)
            eligible, nearest = self.eligible_candidates(view, context, candidates)
            if len(eligible) == 1:
                chosen = eligible[0]
            else:
                probabilities = self.transition_probabilities(view, x, eligible)
                chosen = int(rng.choice(eligible, p=probabilities))
            self.diagnostics.ring_steps[nearest] += 1
            self.diagnostics.sector_distance_sum += nearest
            self.diagnostics.sector_distance_max = max(self.diagnostics.sector_distance_max, nearest)
            self.diagnostics.sb_steps += 1
            max_weight = max(view.cumulative_weight(candidate) for candidate in candidates)
            # This diagnostic measures the locality filter, not the subsequent
            # stochastic draw.  The old comparison against ``chosen`` counted
            # ordinary WRW randomness as if the sector filter had caused it.
            if all(view.cumulative_weight(candidate) < max_weight for candidate in eligible):
                self.diagnostics.mcmc_mode_divergence_steps += 1
            x = chosen
        self.diagnostics.step_limit_exceeded += 1
        raise WalkStepLimitExceeded(self.entry_point(view, context))

    def _endpoint_fallback(self, view: TangleView, context: SelectionContext, rng: np.random.Generator, excluded: frozenset[int]) -> int | None:
        tips = [tip for tip in view.tips(context.required_branch) if tip not in excluded]
        if not tips:
            return None
        eligible, _ = self.eligible_candidates(view, context, tips)
        probabilities = stable_softmax([self.alpha * view.cumulative_weight(y) for y in eligible])
        return int(rng.choice(eligible, p=probabilities))


def normative_sb_wrw_walk(
    view: TangleView,
    issuer_sector: int,
    entry_point: int,
    alpha: float,
    rng: np.random.Generator,
    grid: tuple[int, int],
    max_walk_steps: int = 10000,
    excluded_tips: frozenset[int] = frozenset(),
    required_branch: str | None = None,
) -> int:
    """Clear test oracle corresponding directly to section 5.3.6."""
    x = entry_point
    for step in range(max_walk_steps):
        candidates = tuple(view.direct_approvers(x, required_branch))
        if not candidates:
            return terminal_outcome(view, x, step, excluded_tips, required_branch).endpoint
        nearest = min(sector_distance(issuer_sector, view.sector(y), grid) for y in candidates)
        eligible = [y for y in candidates if sector_distance(issuer_sector, view.sector(y), grid) == nearest]
        probabilities = wrw_transition_probabilities(view, x, eligible, alpha)
        x = int(rng.choice(eligible, p=probabilities))
    raise WalkStepLimitExceeded(entry_point)
