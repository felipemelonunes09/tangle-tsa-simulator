from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class EventKind(IntEnum):
    ATTACK_START = 0
    ATTACH = 1
    ISSUE = 2
    ATTACK_ISSUE = 3
    CONFIDENCE_CHECK = 4
    CONFIDENCE_STABILITY_DEADLINE = 5


@dataclass(order=True, slots=True)
class Event:
    time: float
    counter: int
    kind: EventKind = field(compare=False)
    payload: Any = field(compare=False, default=None)


@dataclass(frozen=True, slots=True)
class PendingAttachment:
    issue_time: float
    issuer_id: int
    parents: tuple[int, ...]
    honest: bool = True
    visible_time: float | None = None
    conflict_id: str | None = None
    branch_id: str | None = None
