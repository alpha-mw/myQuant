"""Opaque immutable snapshot mechanics; factory admission is tested separately."""

from dataclasses import FrozenInstanceError
import hashlib
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.completed_handoff_snapshot import (
    CompletedHandoffSnapshot,
    _mint_snapshot,
    _require_snapshot,
)
from quant_investor.operations.daily_contract import ContractError

ROLES = (
    "completion",
    "ledger",
    "materialization",
    "handoff",
    "recipe",
    "loop_context",
    "release_install_input",
    "logical_claim",
)


def fixture(root):
    docs = []
    for role in ROLES:
        raw = canonical_json_bytes({"role": role})
        path = root / (role + ".json")
        path.write_bytes(raw)
        path.chmod(0o600)
        docs.append((role, path.name, hashlib.sha256(raw).hexdigest(), raw))
    return _mint_snapshot(workspace=str(root), trade_date="20260908", documents=tuple(docs))


def test_snapshot_cannot_be_publicly_constructed_or_mutated(tmp_path):
    with pytest.raises(ContractError, match="FACTORY_REQUIRED"):
        CompletedHandoffSnapshot()
    with pytest.raises(ContractError, match="CAPABILITY_REQUIRED"):
        _require_snapshot({})
    with pytest.raises(ContractError, match="CAPABILITY_REQUIRED"):
        _require_snapshot(object.__new__(CompletedHandoffSnapshot))
    snapshot = fixture(tmp_path)
    snapshot.recheck()
    doc = snapshot.document("handoff")
    doc["role"] = "forged"
    assert snapshot.document("handoff") == {"role": "handoff"}
    with pytest.raises(FrozenInstanceError):
        snapshot.workspace = "elsewhere"
    assert isinstance(snapshot.documents, tuple)


@pytest.mark.parametrize("role", ROLES)
def test_every_captured_file_is_rechecked(tmp_path, role):
    snapshot = fixture(tmp_path)
    (tmp_path / (role + ".json")).write_bytes(b"changed")
    with pytest.raises(ContractError, match="SNAPSHOT_CHANGED"):
        snapshot.recheck()


@pytest.mark.parametrize("state", ["RUNNING", "FAILED", "PARTIAL", None, "MISSING_NODE"])
def test_incomplete_completion_cannot_mint_snapshot(tmp_path, monkeypatch, state):
    from quant_investor.operations import completion_readback as reader
    from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY
    from quant_investor.operations.daily_contract import GRAPH_SHA256, EOD_NODE_IDS

    j = DailyJournal(str(tmp_path), "20260908")
    ref = {"path": "unused.json", "sha256": "a" * 64}
    value = dict(
        schema_version="cn-daily-eod-completion.v2",
        status="SUCCEEDED" if state == "MISSING_NODE" else state,
        market="CN",
        strategy_id="aggressive_tech_manufacturing",
        trade_date="20260908",
        graph_sha256=GRAPH_SHA256,
        release_ref=ref,
        native_inputs_ref=ref,
        node_terminal_refs={} if state == "MISSING_NODE" else {n: ref for n in EOD_NODE_IDS},
        synthetic=True,
        prospective_admission_state="LEDGER_INELIGIBLE",
        authority=FALSE_AUTHORITY,
        native_validation_completed_at="2026-09-08T00:00:00Z",
        materialization_ref=ref,
        prospective_ledger_ref=ref,
    )
    raw = canonical_json_bytes(value)
    stored = j.storage.write(str(j.root / "completion.v1.json"), raw)
    monkeypatch.setattr(
        reader,
        "_completed_handoff_snapshot",
        lambda **kw: pytest.fail("minted from incomplete binding"),
    )
    with pytest.raises(ContractError, match="CONTRACT_INVALID"):
        reader.inspect_recorded_completion(
            workspace=str(tmp_path),
            trade_date="20260908",
            completion_ref={"path": stored.relative_path, "sha256": stored.byte_sha256},
        )
