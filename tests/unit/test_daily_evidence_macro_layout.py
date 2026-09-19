"""Exact daily transaction selection preserves legacy evidence and rejects ambiguity."""

from types import SimpleNamespace
import pytest
from quant_investor.market.daily_macro_layout import select_macro_layout


def context(root):
    return SimpleNamespace(workspace_root=root, run_root=root, target_date="20260819")


@pytest.mark.parametrize(
    "state", ["FRESH", "PREPARED", "JOURNALED", "LEGACY", "partial", "empty", "conflict"]
)
def test_transaction_occupancy_matrix(tmp_path, state):
    ctx = context(tmp_path)
    layout = select_macro_layout(ctx)
    assert len(layout.transaction_id) == 79
    assert layout.journal_id == layout.transaction_id
    assert (
        layout.prepared_path
        == layout.preparation_parent / layout.transaction_id / "prepared/prepared.json"
    )
    transaction = layout.preparation_parent / layout.transaction_id
    journal = layout.journal_root / layout.journal_id
    if state == "PREPARED":
        layout.prepared_path.parent.mkdir(parents=True)
        layout.prepared_path.write_text("{}")
    elif state in {"JOURNALED", "empty"}:
        journal.mkdir(parents=True)
        if state == "JOURNALED":
            (journal / "0001-intent.json").write_text("{}")
    elif state == "partial":
        transaction.mkdir(parents=True)
    elif state in {"LEGACY", "conflict"}:
        legacy = tmp_path / "journals/macro/20260819/macro-20260819"
        legacy.mkdir(parents=True)
        (legacy / "0001-intent.json").write_text("{}")
        if state == "conflict":
            transaction.mkdir(parents=True)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    if state in {"partial", "empty", "conflict"}:
        with pytest.raises(RuntimeError):
            select_macro_layout(ctx)
    else:
        assert select_macro_layout(ctx).state == state
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_identity_is_attempt_independent_and_path_bound(tmp_path):
    ctx = context(tmp_path)
    first = select_macro_layout(ctx)
    ctx.attempt_slot = "another"
    assert select_macro_layout(ctx) == first
    other = tmp_path / "other"
    other.mkdir()
    ctx.run_root = other
    assert select_macro_layout(ctx).transaction_id != first.transaction_id


def test_symlink_transaction_root_is_rejected(tmp_path):
    layout = select_macro_layout(context(tmp_path))
    target = tmp_path / "elsewhere"
    target.mkdir()
    layout.preparation_parent.parent.mkdir(parents=True)
    layout.preparation_parent.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="PATH_UNSAFE"):
        select_macro_layout(context(tmp_path))
