from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .conflicts import branches_compatible, inherited_branch, parent_branches_compatible
from .transaction import Transaction


class TangleInvariantError(ValueError):
    pass


class InconsistentViewError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TangleView:
    tangle: "Tangle"
    cutoff_time: float
    required_branch: str | None = None
    _visible_cache: dict[str | None, tuple[int, ...]] = field(default_factory=dict, compare=False, repr=False)
    _tips_cache: dict[str | None, tuple[int, ...]] = field(default_factory=dict, compare=False, repr=False)
    _cw_cache: dict[int, float] = field(default_factory=dict, compare=False, repr=False)

    def is_visible(self, tx_id: int) -> bool:
        tx = self.tangle.transactions[tx_id]
        return tx.visible_time <= self.cutoff_time

    def is_valid(self, tx_id: int) -> bool:
        tx = self.tangle.transactions[tx_id]
        return tx.honest or tx.visible_time <= self.cutoff_time

    def is_branch_compatible(self, tx_id: int, required_branch: str | None = None) -> bool:
        required = self.required_branch if required_branch is None else required_branch
        return branches_compatible(self.tangle.transactions[tx_id].branch_id, required)

    def direct_approvers(self, tx_id: int, required_branch: str | None = None) -> tuple[int, ...]:
        return tuple(
            child for child in self.tangle.transactions[tx_id].direct_approvers
            if self.is_visible(child)
            and self.is_valid(child)
            and self.is_branch_compatible(child, required_branch)
        )

    def is_tip(self, tx_id: int, required_branch: str | None = None) -> bool:
        required = self.required_branch if required_branch is None else required_branch
        if self.tangle.max_visible_time <= self.cutoff_time:
            return tx_id in self.tangle.branch_tips(required)
        return self.is_visible(tx_id) and not self.direct_approvers(tx_id, required_branch)

    def visible_ids(self, required_branch: str | None = None) -> tuple[int, ...]:
        required = self.required_branch if required_branch is None else required_branch
        if required in self._visible_cache:
            return self._visible_cache[required]
        values = tuple(
            tx.id for tx in self.tangle.transactions
            if self.is_visible(tx.id) and self.is_valid(tx.id)
            and self.is_branch_compatible(tx.id, required)
        )
        self._visible_cache[required] = values
        return values

    def tips(self, required_branch: str | None = None) -> tuple[int, ...]:
        required = self.required_branch if required_branch is None else required_branch
        if required in self._tips_cache:
            return self._tips_cache[required]
        if self.tangle.max_visible_time <= self.cutoff_time:
            values = tuple(sorted(self.tangle.branch_tips(required)))
        else:
            values = tuple(tx_id for tx_id in self.visible_ids(required) if self.is_tip(tx_id, required))
        self._tips_cache[required] = values
        return values

    def cumulative_weight(self, tx_id: int) -> float:
        # Invisible descendants have not contributed to the snapshot yet.
        if self.tangle.max_visible_time <= self.cutoff_time:
            return self.tangle.transactions[tx_id].cumulative_weight
        if tx_id in self._cw_cache:
            return self._cw_cache[tx_id]
        tx = self.tangle.transactions[tx_id]
        total = tx.own_weight
        bit = 1 << tx_id
        for descendant in self.tangle.transactions[tx_id + 1:]:
            if descendant.visible_time <= self.cutoff_time and descendant.past_mask & bit:
                total += descendant.own_weight
        self._cw_cache[tx_id] = total
        return total

    def sector(self, tx_id: int) -> int:
        return self.tangle.transactions[tx_id].sector_id

    def position(self, tx_id: int) -> tuple[float, float]:
        return self.tangle.transactions[tx_id].position

    def branch(self, tx_id: int) -> str | None:
        return self.tangle.transactions[tx_id].branch_id

    def approves(self, tip_id: int, ancestor_id: int) -> bool:
        return bool(self.tangle.transactions[tip_id].past_mask & (1 << ancestor_id))


class Tangle:
    """Compact DAG; a new transaction points to older parents it approves.

    Walks toward tips traverse ``direct_approvers`` in the future direction.
    ``past_mask`` contains the transaction itself and every ancestor, allowing
    exact diamond de-duplication.
    """

    def __init__(self) -> None:
        genesis = Transaction(
            id=0,
            issue_time=0.0,
            attachment_time=0.0,
            visible_time=0.0,
            issuer_id=-1,
            sector_id=0,
            position=(0.5, 0.5),
            parents=(),
            past_mask=1,
        )
        self.transactions: list[Transaction] = [genesis]
        self.tips: set[int] = {0}
        self._tips_by_branch: dict[str | None, set[int]] = {None: self.tips}
        self.attachment_times: list[float] = [0.0]
        self.max_visible_time = 0.0

    def __len__(self) -> int:
        return len(self.transactions)

    def view(self, cutoff_time: float, required_branch: str | None = None) -> TangleView:
        return TangleView(self, float(cutoff_time), required_branch)

    def branch_tips(self, required_branch: str | None) -> set[int]:
        if required_branch not in self._tips_by_branch:
            compatible = [
                tx.id for tx in self.transactions
                if branches_compatible(tx.branch_id, required_branch)
            ]
            compatible_set = set(compatible)
            tips = {
                tx_id for tx_id in compatible
                if not any(child in compatible_set for child in self.transactions[tx_id].direct_approvers)
            }
            self._tips_by_branch[required_branch] = tips
        return self._tips_by_branch[required_branch]

    def add_transaction(
        self,
        *,
        issue_time: float,
        attachment_time: float,
        visible_time: float,
        issuer_id: int,
        sector_id: int,
        position: tuple[float, float],
        parents: Iterable[int],
        own_weight: float = 1.0,
        honest: bool = True,
        conflict_id: str | None = None,
        branch_id: str | None = None,
    ) -> int:
        parent_tuple = tuple(int(p) for p in parents)
        if len(parent_tuple) > 2:
            raise TangleInvariantError("a transaction may approve at most two parents")
        if len(set(parent_tuple)) != len(parent_tuple):
            raise TangleInvariantError("duplicate parent edge")
        if not parent_tuple and self.transactions:
            raise TangleInvariantError("only genesis may have no parent")
        if attachment_time < issue_time or visible_time < attachment_time:
            raise TangleInvariantError("issue <= attachment <= visibility is required")
        if own_weight <= 0:
            raise TangleInvariantError("own_weight must be positive")
        for parent in parent_tuple:
            if parent < 0 or parent >= len(self.transactions):
                raise TangleInvariantError(f"unknown parent: {parent}")
            if self.transactions[parent].attachment_time > attachment_time:
                raise TangleInvariantError("parents must be temporally earlier")
        if attachment_time < self.attachment_times[-1]:
            raise TangleInvariantError("transactions must be attached in event-time order")
        parent_branches = [self.transactions[p].branch_id for p in parent_tuple]
        if not parent_branches_compatible(parent_branches):
            raise TangleInvariantError("incompatible conflict branches in parents")
        inherited = inherited_branch(parent_branches)
        if branch_id is None:
            branch_id = inherited
        elif inherited is not None and branch_id != inherited:
            raise TangleInvariantError("explicit branch conflicts with parent branch")
        if branch_id is not None:
            self.branch_tips(branch_id)

        tx_id = len(self.transactions)
        ancestor_mask = 0
        for parent in parent_tuple:
            ancestor_mask |= self.transactions[parent].past_mask
        mask = ancestor_mask
        while mask:
            lsb = mask & -mask
            ancestor = lsb.bit_length() - 1
            self.transactions[ancestor].cumulative_weight += own_weight
            mask ^= lsb
        tx = Transaction(
            id=tx_id,
            issue_time=float(issue_time),
            attachment_time=float(attachment_time),
            visible_time=float(visible_time),
            issuer_id=int(issuer_id),
            sector_id=int(sector_id),
            position=(float(position[0]), float(position[1])),
            parents=parent_tuple,
            own_weight=float(own_weight),
            cumulative_weight=float(own_weight),
            honest=bool(honest),
            conflict_id=conflict_id,
            branch_id=branch_id,
            past_mask=ancestor_mask | (1 << tx_id),
        )
        self.transactions.append(tx)
        self.attachment_times.append(float(attachment_time))
        self.max_visible_time = max(self.max_visible_time, float(visible_time))
        for parent in parent_tuple:
            parent_tx = self.transactions[parent]
            parent_tx.direct_approvers.append(tx_id)
            if parent_tx.first_direct_approval_time is None:
                parent_tx.first_direct_approval_time = float(attachment_time)
            self.tips.discard(parent)
        self.tips.add(tx_id)
        for required, tips in self._tips_by_branch.items():
            if required is None:
                # The global set is the same object as self.tips and was updated above.
                continue
            if branches_compatible(branch_id, required):
                for parent in parent_tuple:
                    tips.discard(parent)
                tips.add(tx_id)
        return tx_id

    def slow_cumulative_weight(self, tx_id: int, cutoff_time: float | None = None) -> float:
        cutoff = float("inf") if cutoff_time is None else cutoff_time
        bit = 1 << tx_id
        return sum(
            tx.own_weight for tx in self.transactions
            if tx.visible_time <= cutoff and tx.past_mask & bit
        )

    def assert_invariants(self) -> None:
        for tx in self.transactions:
            if tx.id == 0:
                continue
            if len(set(tx.parents)) != len(tx.parents):
                raise TangleInvariantError("duplicate parents")
            for parent in tx.parents:
                if parent >= tx.id:
                    raise TangleInvariantError("cycle or non-past parent")
                if tx.id not in self.transactions[parent].direct_approvers:
                    raise TangleInvariantError("missing reverse edge")
            expected = self.slow_cumulative_weight(tx.id)
            if abs(expected - tx.cumulative_weight) > 1e-12:
                raise TangleInvariantError("incremental cumulative weight mismatch")
