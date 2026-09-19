"""Ledger composition with controlled native replay; not native integration proof."""

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError, GRAPH_SHA256
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY
from scripts import daily_ledger as module
from scripts.daily_materialization import MaterializedInputs
from test_daily_evidence_daily_timing import context as rows_context


def context(root, monkeypatch, version=1, historical=False):
    def put(path, value):
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        current = p.parent
        while current != root:
            current.chmod(0o700)
            current = current.parent
        raw = canonical_json_bytes(value)
        p.write_bytes(raw)
        p.chmod(0o600)
        return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}

    clock = "2026-09-08T00:00:00Z"
    rows = rows_context()["nodes"]
    for row in rows.values():
        row["start"]["started_at"] = clock
        row["terminal"]["finished_at"] = clock
        row["terminal"]["recovered"] = False
    for alias, node in [("LOW", "low_observation"), ("W80", "w80_observation")]:
        ref = put(
            alias + ".json",
            {"payload": {"factor_alias": alias, "signal_date": "20260908", "registered_at": clock}},
        )
        rows[node]["terminal"]["output_refs"] = {alias: ref}
    pointer = put("pointer.json", {"activated_at": clock})
    release = put("release.json", {})
    request = put(
        "research.json", {"industry_source": None, "theme_source": None, "company_evidence": {}}
    )
    other = put("other.json", {})
    inputs = SimpleNamespace(
        release_ref=release,
        research_request_ref=request,
        store_plan_ref=other,
        calendar_ref=other,
        market_snapshot_ref=other,
        factor_pointer_sha256=pointer["sha256"],
    )
    recipe = {
        "schema_version": f"cn-daily-execute-recipe.v{version}",
        "policy_refs": {"prospective": None},
        "retrospective_ref": None,
    }
    if version == 2:
        recipe["theme_acquisition_ref"] = None
    handoff = dict(
        schema_version=(
            "cn-daily-maintenance-handoff.v3" if historical else "cn-daily-maintenance-handoff.v2"
        ),
        recipe_ref=put("recipe.json", recipe),
        trade_date="20260908",
        release_ref=release,
        calendar_ref=other,
        factor_pointer_ref=pointer,
        market_snapshot_ref=other,
        prospective_policy_ref=None,
        sealed_at=clock,
    )
    journal = DailyJournal(str(root), "20260908")
    directory = str(journal.root / "executions" / ("a" * 64))
    handoff_ref = put(directory + "/maintenance-handoff.v1.json", handoff)
    from quant_investor.operations.native_input_contract import FIELDS

    # Native business admission remains an explicit seam; retain its exact legacy
    # envelope so version dispatch is real rather than an empty placeholder.
    native_document = dict.fromkeys(FIELDS)
    native_document.update(
        schema_version="cn-daily-native-inputs.v1", trade_date="20260908", **vars(inputs)
    )
    native = put(directory + "/inputs/native.json", native_document)
    record = dict(
        schema_version=f"cn-daily-materialization.v{version}",
        trade_date="20260908",
        graph_sha256=GRAPH_SHA256,
        maintenance_handoff_ref=handoff_ref,
        research_request_ref=request,
        store_plan_ref=other,
        native_inputs_ref=native,
        auxiliary_stage_refs={"fundamental": None, "macro": None},
        sealed_at=clock,
        authority=FALSE_AUTHORITY,
    )
    if version == 2:
        record["theme_source_handoff_ref"] = None
    materialized = MaterializedInputs(
        put(directory + f"/materialization.v{version}.json", record), native, inputs, "MATERIALIZED"
    )
    registry = SimpleNamespace(
        workspace=str(root),
        trade_date="20260908",
        inputs=inputs,
        runner=SimpleNamespace(journal=journal, _request=lambda template, node, completed: node),
        templates={n: {} for n in rows},
        core=SimpleNamespace(
            pointer_ref=pointer,
            snapshot=lambda: {"factor_generation": {"created_at": "2020-01-01T00:00:00Z"}},
        ),
    )
    monkeypatch.setattr(module, "_validate_native_context", lambda *a, **kw: None)
    monkeypatch.setattr(module, "verify_loaded_native_inputs", lambda **kw: None)
    monkeypatch.setattr(module, "verify_materialized_inputs", lambda **kw: None)
    monkeypatch.setattr(
        module, "_replay", lambda registry: {n: r["terminal_ref"] for n, r in rows.items()}
    )
    monkeypatch.setattr(
        module,
        "read_maintenance_handoff",
        lambda **kw: {
            "handoff": handoff,
            "recipe": recipe,
        },
    )
    monkeypatch.setattr(journal, "inspect", lambda node: rows[node])
    return registry, materialized


def test_assembly_binds_selected_evidence_and_never_publishes(tmp_path, monkeypatch):
    registry, materialized = context(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with registry.runner.journal.locked():
        result = module.assemble_native_ledger(registry, materialized=materialized, synthetic=True)
    assert result["classification"] == "RETROSPECTIVE_RECOMPUTE"
    assert result["prospective"] is False and result["recomputed"] is True
    assert result["materialization_ref"] == materialized.materialization_ref
    assert result["native_inputs_ref"] == materialized.native_inputs_ref
    assert len(result["node_terminal_refs"]) == 16
    assert result["core_timing"]["generation_created_at_is_availability_proof"] is False
    assert not (tmp_path / "results/prospective").exists()
    for path, raw in before.items():
        assert Path(path).read_bytes() == raw


def test_assembly_requires_existing_lock(tmp_path, monkeypatch):
    registry, materialized = context(tmp_path, monkeypatch)
    with pytest.raises(ContractError, match="JOURNAL_LOCK_REQUIRED"):
        module.assemble_native_ledger(registry, materialized=materialized, synthetic=True)


def test_published_ledger_adopts_identical_bytes_after_interruption(tmp_path, monkeypatch):
    from quant_investor.operations.prospective_storage import ProspectiveLedgerStorage

    registry, materialized = context(tmp_path, monkeypatch)
    with registry.runner.journal.locked():
        first = module.publish_native_ledger(registry, materialized=materialized, synthetic=True)
    path = tmp_path / first["ledger_ref"]["path"]
    original = path.read_bytes()
    identity = path.stat().st_ino
    assert first["status"] == "PUBLISHED"
    assert not (tmp_path / registry.runner.journal.root / "completion.v1.json").exists()
    monkeypatch.setattr(
        ProspectiveLedgerStorage, "write", lambda *a, **kw: pytest.fail("rewrote existing ledger")
    )
    with registry.runner.journal.locked():
        again = module.publish_native_ledger(registry, materialized=materialized, synthetic=True)
    assert again["status"] == "NO_ACTION"
    assert again["ledger_ref"] == first["ledger_ref"]
    assert again["ledger"]["published_at"] == first["ledger"]["published_at"]
    assert path.read_bytes() == original and path.stat().st_ino == identity


@pytest.mark.parametrize("fault", ["classification", "future_clock"])
def test_conflicting_ledger_is_never_overwritten(tmp_path, monkeypatch, fault):
    import json

    registry, materialized = context(tmp_path, monkeypatch)
    with registry.runner.journal.locked():
        first = module.publish_native_ledger(registry, materialized=materialized, synthetic=True)
    path = tmp_path / first["ledger_ref"]["path"]
    changed = json.loads(path.read_bytes())
    if fault == "classification":
        changed["prospective"] = True
    else:
        changed["published_at"] = "2999-01-01T00:00:00Z"
    path.write_bytes(canonical_json_bytes(changed))
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with registry.runner.journal.locked(), pytest.raises(ContractError):
        module.publish_native_ledger(registry, materialized=materialized, synthetic=True)
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_occupied_legacy_eod_blocks_before_ledger_creation(tmp_path, monkeypatch):
    registry, materialized = context(tmp_path, monkeypatch)
    journal = registry.runner.journal
    with journal.locked():
        journal.storage.write(
            str(journal.root / "completion.v1.json"),
            canonical_json_bytes({"schema_version": "cn-daily-eod-completion.v1"}),
        )
        monkeypatch.setattr(
            module,
            "assemble_native_ledger",
            lambda *a, **kw: pytest.fail("assembled behind legacy EOD"),
        )
        with pytest.raises(ContractError, match="EOD_LEGACY_PATH_OCCUPIED"):
            module.publish_native_ledger(registry, materialized=materialized, synthetic=True)
    assert not (tmp_path / "results/prospective").exists()
