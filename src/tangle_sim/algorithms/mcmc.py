from __future__ import annotations

import numpy as np

from tangle_sim.domain import TangleView
from .base import (
    ExcludedEndpointReached,
    RandomWalkSelector,
    SelectionContext,
    WalkOutcome,
    WalkStepLimitExceeded,
    terminal_outcome,
    wrw_transition_probabilities,
)


class MCMC(RandomWalkSelector):
    name = "mcmc"

    def transition_probabilities(
        self,
        view: TangleView,
        current: int,
        candidates: tuple[int, ...] | list[int],
    ) -> np.ndarray:
        return wrw_transition_probabilities(view, current, candidates, self.alpha)

    def walk(self, view: TangleView, context: SelectionContext, rng: np.random.Generator, excluded_tips: frozenset[int] = frozenset()) -> WalkOutcome:
        x = self.entry_point(view, context)
        for step in range(self.max_walk_steps):
            candidates = view.direct_approvers(x, context.required_branch)
            if not candidates:
                return terminal_outcome(view, x, step, excluded_tips, context.required_branch)
            if len(candidates) == 1:
                x = candidates[0]
            else:
                probabilities = self.transition_probabilities(view, x, candidates)
                x = int(rng.choice(candidates, p=probabilities))
        self.diagnostics.step_limit_exceeded += 1
        raise WalkStepLimitExceeded(self.entry_point(view, context))
