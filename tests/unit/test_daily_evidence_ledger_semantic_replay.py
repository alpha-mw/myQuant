"""Semantic reconstruction with explicit controlled native validation seams."""

import hashlib, json
from copy import deepcopy
from types import SimpleNamespace
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.journal_storage import JournalStorage
from test_daily_evidence_ledger_assembly import context
from scripts import daily_ledger as producer
from scripts import daily_ledger_replay as consumer


@pytest.mark.parametrize("fault", [None, "core", "sources", "classification"])
@pytest.mark.parametrize("already_loaded", [False, True])
@pytest.mark.parametrize("historical", [False, True])
def test_semantic_replay_recomputes_from_native_inputs(
    tmp_path, monkeypatch, fault, already_loaded, historical
):
    registry, materialized = context(tmp_path, monkeypatch, historical=historical)
    with registry.runner.journal.locked():
        published = producer.publish_native_ledger(
            registry, materialized=materialized, synthetic=not historical
        )
    assert published["ledger"]["recomputed"] is True
    original = deepcopy(published["ledger"])
    path = tmp_path / published["ledger_ref"]["path"]
    document = json.loads(path.read_bytes())
    if fault == "core":
        document["core_timing"]["generation_created_at"] = "2021-01-01T00:00:00Z"
    elif fault == "sources":
        document["source_times"]["macro"] = [
            {
                "source_ref": {"path": "forged.json", "sha256": "a" * 64},
                "declared_available_at": None,
            }
        ]
    elif fault == "classification":
        document["classification"] = "CONTEMPORANEOUS"
    raw = canonical_json_bytes(document)
    path.write_bytes(raw)
    ledger_ref = {**published["ledger_ref"], "sha256": hashlib.sha256(raw).hexdigest()}
    completion = {
        "schema_version": "cn-daily-eod-completion.v2",
        "prospective_ledger_ref": ledger_ref,
        "materialization_ref": materialized.materialization_ref,
        "native_inputs_ref": materialized.native_inputs_ref,
        "synthetic": not historical,
    }
    inspected = {
        "recorded_completion": completion,
        "recorded_custody_timing": original["node_custody"],
        "completed_handoff_snapshot": SimpleNamespace(document=lambda role: deepcopy(document)),
    }
    monkeypatch.setattr(consumer, "inspect_recorded_completion", lambda **kw: deepcopy(inspected))
    monkeypatch.setattr(
        consumer,
        "read_recorded_maintenance_handoff",
        lambda snapshot: producer.read_maintenance_handoff(),
    )
    monkeypatch.setattr(consumer, "load_native_inputs", lambda **kw: ("20260908", registry.inputs))
    monkeypatch.setattr(consumer, "verify_materialized_inputs", lambda **kw: None)
    monkeypatch.setattr(
        consumer,
        "replay_completed_core",
        lambda **kw: {"recorded_timing": deepcopy(original["core_timing"])},
    )

    def forbidden(*a, **kw):
        pytest.fail("semantic replay used a writer or lock")

    monkeypatch.setattr(DailyJournal, "locked", forbidden)
    monkeypatch.setattr(JournalStorage, "write", forbidden)
    args = dict(
        workspace=str(tmp_path),
        trade_date="20260908",
        completion_ref={"path": "controlled.json", "sha256": "a" * 64},
    )
    if already_loaded:
        args["loaded_inputs"] = registry.inputs
        monkeypatch.setattr(consumer, "load_native_inputs", forbidden)
        monkeypatch.setattr(consumer, "verify_loaded_native_inputs", lambda **kw: None)
    if fault is None:
        result = consumer.replay_completed_ledger(**args)
        assert result["prospective"] is False
        assert result["classification"] == "RETROSPECTIVE_RECOMPUTE"
        assert result["native_business_replay_required"] is True
    else:
        with pytest.raises(ContractError, match="DERIVATION_MISMATCH"):
            consumer.replay_completed_ledger(**args)
