from __future__ import annotations

from collections.abc import Iterable


def branches_compatible(branch_a: str | None, branch_b: str | None) -> bool:
    return branch_a is None or branch_b is None or branch_a == branch_b


def parent_branches_compatible(branches: Iterable[str | None]) -> bool:
    concrete = {branch for branch in branches if branch is not None}
    return len(concrete) <= 1


def inherited_branch(branches: Iterable[str | None]) -> str | None:
    concrete = {branch for branch in branches if branch is not None}
    if len(concrete) > 1:
        raise ValueError("parents belong to incompatible conflict branches")
    return next(iter(concrete), None)

