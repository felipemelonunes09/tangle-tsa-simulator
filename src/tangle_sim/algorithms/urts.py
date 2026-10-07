from __future__ import annotations

from dataclasses import replace

import numpy as np

from tangle_sim.domain import TangleView
from .base import SelectionContext, SelectorDiagnostics


class URTS:
    name = "urts"

    def __init__(self) -> None:
        self.diagnostics = SelectorDiagnostics()

    def select_tip(self, view: TangleView, context: SelectionContext, rng: np.random.Generator, excluded_tips: frozenset[int] = frozenset()) -> int:
        tips = [
            tip for tip in view.tips(context.required_branch)
            if tip not in excluded_tips
            and (
                context.required_ancestor is None
                or tip == context.required_ancestor
                or view.approves(tip, context.required_ancestor)
            )
        ]
        if not tips:
            raise RuntimeError("no compatible visible tip")
        return int(rng.choice(tips))

    def select_parents(self, view: TangleView, context: SelectionContext, rng: np.random.Generator) -> tuple[int, ...]:
        self.diagnostics.selections += 1
        first = self.select_tip(view, context, rng)
        branch = view.branch(first)
        second_context = replace(context, required_branch=branch)
        eligible = [
            tip for tip in view.tips(branch)
            if tip != first
            and (
                context.required_ancestor is None
                or tip == context.required_ancestor
                or view.approves(tip, context.required_ancestor)
            )
        ]
        if branch is not None:
            self.diagnostics.compatibility_restrictions += 1
        if not eligible:
            self.diagnostics.single_parent += 1
            parents = (first,)
        else:
            parents = (first, int(rng.choice(eligible)))
        for endpoint in parents:
            self.diagnostics.endpoint_sectors[view.sector(endpoint)] += 1
        return parents
