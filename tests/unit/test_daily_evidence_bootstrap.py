"""Bootstrap preserves exact native predecessor and multi-day Store coverage."""

import hashlib
from types import SimpleNamespace
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations import bootstrap as module
from quant_investor.operations.daily_contract import GRAPH_SHA256, ContractError
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY


def fixture(root, monkeypatch):
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

    parent_raw = canonical_json_bytes({"fixture": "registered parent"})
    sha = hashlib.sha256(parent_raw).hexdigest()
    parent = put(
        str(module.FACTOR_POINTER_HISTORY_ROOT / f"{sha}.json"), {"fixture": "registered parent"}
    )
    target = put("target.json", {"previous_pointer_sha256": sha})
    other = put("source.json", {})
    preimages = {
        name: other for name in ("store_pointer_ref", "event_pointer_ref", "benchmark_pointer_ref")
    }
    declaration = dict(
        schema_version="cn-daily-bootstrap.v1",
        market="CN",
        strategy_id="aggressive_tech_manufacturing",
        first_trade_date="20260827",
        previous_trade_date="20260826",
        graph_sha256=GRAPH_SHA256,
        factor_parent_pointer_sha256=sha,
        store_preimages=preimages,
        authority=FALSE_AUTHORITY,
    )
    recipe = dict(
        target_trade_date="20260827",
        bootstrap_ref=put("bootstrap.json", declaration),
        previous_completion_ref=None,
        store_preimages=preimages,
    )
    handoff = dict(calendar_ref=other, raw_calendar_ref=other, factor_pointer_ref=target)
    dates = ["20260821", "20260824", "20260825", "20260826", "20260827"]
    monkeypatch.setattr(
        module,
        "replay_close_session_authority",
        lambda *a: SimpleNamespace(
            receipt={"target_trade_date": "20260827", "ordered_open_dates": dates}
        ),
    )
    calls = []

    def native(**kw):
        assert kw["expected_trade_date"] == "20260827"  # parent is not a research observation
        calls.append(kw["expected_trade_date"])
        assert hashlib.sha256(kw["pointer_raw"]).hexdigest() == kw["expected_pointer_sha256"]
        return {}

    marker = object()

    def baseline(stored, selected_marker):
        assert selected_marker is marker
        assert hashlib.sha256(stored.data).hexdigest() == stored.byte_sha256 == sha
        calls.append("20260826")
        return {"factor_authority": module.FACTOR_AUTHORITY_ACTIVE, "as_of": "20260826"}

    monkeypatch.setattr(
        module,
        "FactorProductionStore",
        lambda *a: SimpleNamespace(
            inspect_recorded_research_inputs=native,
            read=lambda *a: marker,
            _verify_pointer_lineage=baseline,
        ),
    )
    plan = {
        "requested_target": "2026-08-27",
        "last_official_date": "2026-08-21",
        "missing_dates": ["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27"],
        "preimages": {
            key: other["sha256"]
            for key in (
                "store_pointer_sha256",
                "event_pointer_sha256",
                "benchmark_pointer_sha256",
                "calendar_receipt_sha256",
            )
        },
    }
    return dict(recipe=recipe, handoff=handoff), plan, calls, dates


@pytest.mark.parametrize("fault", [None, "inactive", "date", "sha", "marker_race"])
def test_initial_bootstrap_requires_same_native_active_baseline(tmp_path, monkeypatch, fault):
    recovered, _, _, _ = fixture(tmp_path, monkeypatch)
    declaration = module.read_bootstrap_declaration(
        workspace=str(tmp_path), recipe=recovered["recipe"]
    )
    sha = declaration["factor_parent_pointer_sha256"]
    pointer = SimpleNamespace(byte_sha256=sha if fault != "sha" else "b" * 64)
    marker = object()
    changed_marker = object()
    reads = 0
    calls = []

    def read(path):
        nonlocal reads
        if path == module.FACTOR_ACTIVE_POINTER_PATH:
            return pointer
        reads += 1
        return changed_marker if fault == "marker_race" and reads > 1 else marker

    def verify():
        calls.append("native_verify_active")
        return {
            "factor_authority": (
                "INACTIVE" if fault == "inactive" else module.FACTOR_AUTHORITY_ACTIVE
            ),
            "as_of": "20260825" if fault == "date" else "20260826",
            "factor_pointer_byte_sha256": sha,
        }

    monkeypatch.setattr(
        module, "FactorProductionStore", lambda _: SimpleNamespace(read=read, verify_active=verify)
    )
    if fault:
        with pytest.raises(ContractError, match="BOOTSTRAP_INITIAL"):
            module.verify_initial_bootstrap_baseline(
                workspace=str(tmp_path), recipe=recovered["recipe"]
            )
    else:
        result = module.verify_initial_bootstrap_baseline(
            workspace=str(tmp_path), recipe=recovered["recipe"]
        )
        assert result["previous_trade_date"] == "20260826"
        assert result["execution_authorized"] is False
        assert calls == ["native_verify_active"]
    if fault == "sha":
        assert calls == []


def test_native_baseline_and_original_multiday_store_prefix(tmp_path, monkeypatch):
    recovered, plan, calls, _ = fixture(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    proof = module.verify_bootstrap_native(workspace=str(tmp_path), recovered=recovered)
    assert calls == ["20260827", "20260826"]
    module.verify_bootstrap_store_plan(
        proof=proof, plan=plan, calendar_ref=recovered["handoff"]["calendar_ref"]
    )
    assert proof["previous_trade_date"] == "20260826" and proof["execution_authorized"] is False
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    journal = DailyJournal(str(tmp_path), "20260827")
    with journal.locked():
        module.verify_bootstrap_initial_absence(journal=journal, proof=proof)
        journal.storage.write(str(journal.root.parent / "20260826" / "completion.v1.json"), b"{}")
        with pytest.raises(ContractError, match="EXISTING_EOD"):
            module.verify_bootstrap_initial_absence(journal=journal, proof=proof)


@pytest.mark.parametrize(
    "fault", ["calendar", "parent", "bootstrap_sha", "store_target", "store_preimage", "store_gap"]
)
def test_wrong_native_bindings_block(tmp_path, monkeypatch, fault):
    recovered, plan, calls, dates = fixture(tmp_path, monkeypatch)
    if fault == "calendar":
        dates.remove("20260826")
    elif fault == "parent":
        recovered["handoff"]["factor_pointer_ref"] = recovered["handoff"]["calendar_ref"]
    elif fault == "bootstrap_sha":
        (tmp_path / "bootstrap.json").write_bytes(b"changed")
    elif fault == "store_target":
        plan["requested_target"] = "2026-08-28"
    elif fault == "store_preimage":
        plan["preimages"]["store_pointer_sha256"] = "a" * 64
    else:
        plan["missing_dates"].pop(0)
    with pytest.raises(ContractError):
        proof = module.verify_bootstrap_native(workspace=str(tmp_path), recovered=recovered)
        module.verify_bootstrap_store_plan(
            proof=proof, plan=plan, calendar_ref=recovered["handoff"]["calendar_ref"]
        )
