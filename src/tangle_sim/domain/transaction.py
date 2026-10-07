from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Transaction:
    id: int
    issue_time: float
    attachment_time: float
    visible_time: float
    issuer_id: int
    sector_id: int
    position: tuple[float, float]
    parents: tuple[int, ...]
    direct_approvers: list[int] = field(default_factory=list)
    own_weight: float = 1.0
    cumulative_weight: float = 1.0
    honest: bool = True
    conflict_id: str | None = None
    branch_id: str | None = None
    first_direct_approval_time: float | None = None
    confirmed: bool = False
    past_mask: int = 0

    def age(self, at_time: float) -> float:
        return max(0.0, at_time - self.issue_time)

