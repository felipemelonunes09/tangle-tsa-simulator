from __future__ import annotations

import numpy as np
import pytest

from tangle_sim.domain import Tangle, TangleInvariantError


def add(tangle: Tangle, parents: tuple[int, ...], time: float, **kwargs: object) -> int:
    return tangle.add_transaction(
        issue_time=time,
        attachment_time=time,
        visible_time=float(kwargs.pop("visible_time", time)),
        issuer_id=int(kwargs.pop("issuer_id", 0)),
        sector_id=int(kwargs.pop("sector_id", 0)),
        position=kwargs.pop("position", (0.1, 0.1)),
        parents=parents,
        **kwargs,
    )


def test_genesis_and_single_parent_bootstrap() -> None:
    tangle = Tangle()
    assert tangle.tips == {0}
    first = add(tangle, (0,), 1.0)
    assert first == 1
    assert tangle.tips == {1}
    assert tangle.transactions[0].first_direct_approval_time == 1.0


def test_duplicate_parent_and_future_parent_rejected() -> None:
    tangle = Tangle()
    parent = add(tangle, (0,), 2.0)
    with pytest.raises(TangleInvariantError, match="duplicate"):
        add(tangle, (parent, parent), 3.0)
    with pytest.raises(TangleInvariantError, match="temporally"):
        add(tangle, (parent,), 1.0)


def test_diamond_does_not_double_count_cumulative_weight() -> None:
    tangle = Tangle()
    left = add(tangle, (0,), 1.0)
    right = add(tangle, (0,), 1.1)
    diamond = add(tangle, (left, right), 2.0)
    assert diamond == 3
    assert tangle.transactions[0].cumulative_weight == 4.0
    assert tangle.transactions[left].cumulative_weight == 2.0
    assert tangle.slow_cumulative_weight(0) == 4.0
    tangle.assert_invariants()


def test_incremental_matches_slow_on_random_small_dags() -> None:
    rng = np.random.default_rng(44)
    for _ in range(25):
        tangle = Tangle()
        for tx_id in range(1, 40):
            count = 1 if tx_id == 1 or rng.random() < 0.3 else 2
            parents = tuple(int(x) for x in rng.choice(tx_id, size=min(count, tx_id), replace=False))
            add(tangle, parents, float(tx_id))
        for tx in tangle.transactions:
            assert tx.cumulative_weight == tangle.slow_cumulative_weight(tx.id)


def test_snapshot_does_not_see_future_approver() -> None:
    tangle = Tangle()
    parent = add(tangle, (0,), 1.0)
    add(tangle, (parent,), 2.0, visible_time=10.0)
    early = tangle.view(5.0)
    late = tangle.view(10.0)
    assert early.direct_approvers(parent) == ()
    assert early.is_tip(parent)
    assert late.direct_approvers(parent) != ()
    assert early.cumulative_weight(parent) == 1.0
    assert late.cumulative_weight(parent) == 2.0


def test_incompatible_conflict_branches_never_share_child() -> None:
    tangle = Tangle()
    honest = add(tangle, (0,), 1.0, conflict_id="c", branch_id="honest")
    fraud = add(tangle, (0,), 1.0, conflict_id="c", branch_id="fraud", honest=False)
    with pytest.raises(TangleInvariantError, match="incompatible"):
        add(tangle, (honest, fraud), 2.0)


def test_incremental_branch_tip_index_matches_definition() -> None:
    tangle = Tangle()
    neutral = add(tangle, (0,), 0.5)
    honest = add(tangle, (neutral,), 1.0, conflict_id="c", branch_id="honest")
    fraud = add(tangle, (neutral,), 1.1, conflict_id="c", branch_id="fraud", honest=False)
    honest_child = add(tangle, (honest,), 2.0)
    for branch in (None, "honest", "fraud"):
        compatible = [tx.id for tx in tangle.transactions if tx.branch_id is None or branch is None or tx.branch_id == branch]
        expected = {
            tx_id for tx_id in compatible
            if not any(child in compatible for child in tangle.transactions[tx_id].direct_approvers)
        }
        assert tangle.branch_tips(branch) == expected
    assert honest_child in tangle.branch_tips("honest")
    assert fraud in tangle.branch_tips("fraud")
