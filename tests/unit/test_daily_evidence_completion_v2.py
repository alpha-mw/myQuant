"""V2 publication ordering/recovery with explicit controlled native replay seam."""

import json
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import EOD_NODE_IDS, ContractError
from test_daily_evidence_ledger_assembly import context
from scripts import daily_completion as sealer
from scripts import daily_ledger as ledger
from scripts import daily_completion_replay as replay
from scripts import daily_native_inputs as inputs


def setup(root, monkeypatch, fail=False, version=1):
    registry, materialized = context(root, monkeypatch, version)
    monkeypatch.setattr(inputs, "verify_loaded_native_inputs", lambda **kw: None)

    def native_replay(**kw):
        if fail:
            raise ContractError("CONTROLLED_NATIVE_REPLAY_FAILURE")
        doc = json.loads((root / kw["completion_ref"]["path"]).read_bytes())
        return dict(
            native_replay_validated=True,
            completion_ref=kw["completion_ref"],
            trade_date=registry.trade_date,
            synthetic=True,
            validated_nodes=sorted(EOD_NODE_IDS),
            ledger={"ledger_ref": doc["prospective_ledger_ref"]},
        )

    monkeypatch.setattr(replay, "replay_native_completion", native_replay)
    return registry, materialized


@pytest.mark.parametrize("version", [1, 2])
def test_v2_seal_binds_ledger_and_replays_without_republication(tmp_path, monkeypatch, version):
    registry, materialized = setup(tmp_path, monkeypatch, version=version)
    with registry.runner.journal.locked():
        ref = sealer.seal_materialized_completion(
            registry, materialized=materialized, synthetic=True
        )
    doc = json.loads((tmp_path / ref["path"]).read_bytes())
    ledger_doc = json.loads((tmp_path / doc["prospective_ledger_ref"]["path"]).read_bytes())
    assert doc["schema_version"] == "cn-daily-eod-completion.v2"
    assert doc["materialization_ref"] == materialized.materialization_ref
    assert doc["prospective_admission_state"] == "LEDGER_INELIGIBLE"
    assert doc["native_validation_completed_at"] >= ledger_doc["published_at"]
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        ledger, "publish_native_ledger", lambda *a, **kw: pytest.fail("republished ledger")
    )
    with registry.runner.journal.locked():
        assert (
            sealer.seal_materialized_completion(registry, materialized=materialized, synthetic=True)
            == ref
        )
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_legacy_path_blocks_new_v2_without_ledger(tmp_path, monkeypatch):
    registry, materialized = setup(tmp_path, monkeypatch)
    journal = registry.runner.journal
    with journal.locked():
        journal.storage.write(
            str(journal.root / "completion.v1.json"),
            canonical_json_bytes({"schema_version": "cn-daily-eod-completion.v1"}),
        )
        with pytest.raises(ContractError, match="LEGACY_PATH_OCCUPIED"):
            sealer.seal_materialized_completion(registry, materialized=materialized, synthetic=True)
    assert not (tmp_path / "results/prospective").exists()


def test_failed_final_native_replay_never_returns_completion_success(tmp_path, monkeypatch):
    registry, materialized = setup(tmp_path, monkeypatch, fail=True)
    with (
        registry.runner.journal.locked(),
        pytest.raises(ContractError, match="NATIVE_REPLAY_FAILURE"),
    ):
        sealer.seal_materialized_completion(registry, materialized=materialized, synthetic=True)
    # Immutable candidate remains for exact later replay; no rollback or overwrite.
    assert (tmp_path / registry.runner.journal.root / "completion.v1.json").exists()


def test_exact_input_entry_requires_materialization_before_writers(tmp_path):
    from test_daily_evidence_native_inputs import context as input_context, put

    value, args = input_context(tmp_path)
    ref = put(tmp_path, value)
    pointer = args["record_root"] / "_record_store/current.v1.json"
    before = pointer.read_bytes()
    with pytest.raises(ContractError, match="MATERIALIZATION_REQUIRED"):
        sealer.run_materialized_native_input(
            workspace=str(tmp_path), input_ref=ref, resume=True, synthetic=True
        )
    assert pointer.read_bytes() == before
    assert not (tmp_path / "results/prospective").exists()


@pytest.mark.parametrize("version", [1, 2])
def test_exact_materialized_entry_checks_parent_and_keeps_one_decoder(
    tmp_path, monkeypatch, version
):
    from test_daily_evidence_daily_materialization import context as materialization_context
    from scripts.daily_materialization import materialize_locked
    import quant_investor.operations.maintenance_handoff as handoffs

    journal, recovered, book = materialization_context(tmp_path, version)
    with journal.locked():
        materialized = materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
    monkeypatch.setattr(handoffs, "read_maintenance_handoff", lambda **kw: recovered)
    previous = recovered["recipe"]["previous_completion_ref"]
    calls = []

    def parent(**kw):
        assert kw["completion_ref"] == previous
        calls.append("parent")
        return {
            "native_replay_validated": True,
            "completion_ref": previous,
            "trade_date": "20260821",
            "validated_nodes": sorted(EOD_NODE_IDS),
        }

    monkeypatch.setattr(replay, "replay_native_completion", parent)
    decoder = sealer.load_native_inputs

    def once(**kw):
        calls.append("decode")
        assert calls.count("decode") == 1
        return decoder(**kw)

    monkeypatch.setattr(sealer, "load_native_inputs", once)
    result = sealer.run_materialized_native_input(
        workspace=str(tmp_path),
        input_ref=materialized.native_inputs_ref,
        resume=True,
        synthetic=True,
    )
    assert calls == ["decode", "parent"]
    assert result["status"] != "COMPLETE" and result["completion_ref"] is None
