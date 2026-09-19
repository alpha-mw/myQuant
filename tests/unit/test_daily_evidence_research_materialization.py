"""Research-input materialization uses exact refs; source readiness stays native."""

import hashlib
import json
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.intelligence.storage import approved_theme_policy_v2
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.research_materialization import publish_research_input


def context(root, trade_date="20260904"):
    def put(name, value):
        raw = canonical_json_bytes(value)
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        parent = path.parent
        while parent != root:
            parent.chmod(0o700)
            parent = parent.parent
        path.write_bytes(raw)
        path.chmod(0o600)
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}

    journal = DailyJournal(str(root), trade_date)
    policy = put("policy.json", approved_theme_policy_v2())
    low = put("low.json", {"fixture": "LOW"})
    w80 = put("w80.json", {"fixture": "W80"})
    pool = put("pool.json", {"fixture": "pool"})
    outputs = {
        "low_observation": {"LOW": low},
        "w80_observation": {"W80": w80},
        "top100": {"manifest.json": pool},
    }
    nodes = {
        node: put(node + ".json", {"state": "SUCCEEDED", "output_refs": refs})
        for node, refs in outputs.items()
    }
    core = put("core.json", {"node_refs": nodes})
    fundamental_pointer = put("fundamental-pointer.json", {"fixture": "exact pointer"})
    fundamental_daily = put("fundamental-daily.json", {"fixture": "daily ref"})
    fundamental = put(
        "fundamental-source.json",
        {
            "available_at": "2026-09-01T08:00:00Z",
            "pointer": fundamental_pointer,
            "daily_parquet": fundamental_daily,
        },
    )
    macro = put("macro-source.json", {"fixture": "explicit macro source"})
    recipe = {
        "strategy_id": "aggressive_tech_manufacturing",
        "policy_refs": {"research": policy},
        "research_sources": {
            "as_of": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}T13:30:00Z",
            "industry_source_ref": None,
            "theme_source_ref": None,
            "exposure_rows_ref": None,
            "fundamental": {"mode": "PINNED", "source_ref": fundamental},
            "macro": {"mode": "PINNED", "source_ref": macro},
        },
    }
    recipe_ref = put("recipe.json", recipe)
    execution = journal.root / "executions" / ("a" * 64)
    value = {
        "trade_date": trade_date,
        "request_ref": {"path": "request.json", "sha256": "a" * 64},
        "recipe_ref": recipe_ref,
        "core_handoff_ref": core,
        "factor_pointer_ref": {"path": "pointer.json", "sha256": "b" * 64},
    }
    handoff_ref = put(str(execution / "maintenance-handoff.v1.json"), value)
    return journal, {"handoff_ref": handoff_ref, "handoff": value, "recipe": recipe}, put


def test_exact_pinned_sources_and_observations_are_bound_idempotently(tmp_path):
    journal, recovered, _ = context(tmp_path)
    with journal.locked():
        result = publish_research_input(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
        first = (tmp_path / result["research_request_ref"]["path"]).read_bytes()
        again = publish_research_input(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
    assert result == again
    value = json.loads(first)
    assert value["expected_factor_pointer_sha256"] == "b" * 64
    assert value["expected_trade_date"] == "20260904"
    assert value["low_observation_path"] == "low.json"
    assert value["w80_observation_path"] == "w80.json"
    assert value["company_evidence"]["macro_risk"] == {"fixture": "explicit macro source"}
    assert value["industry_source"] is None
    assert result["native_replay_required"] is True and result["execution_authorized"] is False


def test_unavailable_native_stage_stays_missing_without_current_lookup(tmp_path):
    journal, recovered, put = context(tmp_path)
    recipe = recovered["recipe"]
    recipe["research_sources"]["fundamental"] = {"mode": "MAINTENANCE_STAGE", "source_ref": None}
    recipe_ref = put("recipe.json", recipe)
    recovered["handoff"]["recipe_ref"] = recipe_ref
    recovered["handoff_ref"] = put(recovered["handoff_ref"]["path"], recovered["handoff"])
    auxiliary = {"stages": {"fundamental": {"state": "MISSING", "ref": None, "document": None}}}
    with journal.locked():
        result = publish_research_input(journal=journal, recovered=recovered, auxiliary=auxiliary)
    value = json.loads((tmp_path / result["research_request_ref"]["path"]).read_text())
    assert value["company_evidence"]["fundamental_source"] is None
    assert "fundamental:NATIVE_STAGE_MISSING" in result["missing_sources"]
    assert result["auxiliary_stage_refs"]["fundamental"] is None


def test_caller_must_own_day_lock_before_materialization(tmp_path):
    journal, recovered, _ = context(tmp_path)
    with pytest.raises(ContractError, match="LOCK_REQUIRED"):
        publish_research_input(journal=journal, recovered=recovered, auxiliary={"stages": {}})


def test_produced_source_requires_exact_attempt_and_descriptor_ref(tmp_path):
    journal, recovered, put = context(tmp_path)
    recipe = recovered["recipe"]
    recipe["research_sources"]["fundamental"] = {"mode": "MAINTENANCE_STAGE", "source_ref": None}
    recovered["handoff"]["recipe_ref"] = put("recipe.json", recipe)
    core = put("maintenance/attempts/one/core-completion.json", {"fixture": "core"})
    recovered["handoff"]["maintenance_core_ref"] = core
    recovered["handoff_ref"] = put(recovered["handoff_ref"]["path"], recovered["handoff"])
    source_pointer = put("source-pointer.json", {"fixture": "native pointer bytes"})
    source = put(
        "produced-source.json",
        {
            "available_at": "2026-09-04T08:00:00Z",
            "pointer": source_pointer,
            "daily_parquet": source_pointer,
        },
    )
    stage = {
        "state": "STAGE_COMPLETED",
        "result": {
            "stage": "FUNDAMENTAL",
            "status": "READY",
            "blockers": [],
            "evidence": {"research_source_ref": source},
        },
    }
    stage_ref = put("maintenance/attempts/one/stage-FUNDAMENTAL.json", stage)
    auxiliary = {
        "stages": {"fundamental": {"state": "RECORDED", "ref": stage_ref, "document": stage}}
    }
    with journal.locked():
        result = publish_research_input(journal=journal, recovered=recovered, auxiliary=auxiliary)
    document = json.loads((tmp_path / result["research_request_ref"]["path"]).read_text())
    selected = document["company_evidence"]["fundamental_source"]
    assert selected["available_at"] == "2026-09-04T08:00:00Z"
    assert selected["pointer"]["sha256"] == source_pointer["sha256"]
    assert selected["pointer"]["path"] != source_pointer["path"]
    assert result["auxiliary_stage_refs"]["fundamental"] == stage_ref
    auxiliary["stages"]["fundamental"]["ref"] = put(
        "maintenance/attempts/other/stage-FUNDAMENTAL.json", stage
    )
    with journal.locked():
        with pytest.raises(ContractError, match="ATTEMPT_MISMATCH"):
            publish_research_input(journal=journal, recovered=recovered, auxiliary=auxiliary)


def test_retained_fundamental_pointer_survives_current_pointer_change(tmp_path):
    journal, recovered, _ = context(tmp_path)
    with journal.locked():
        first = publish_research_input(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
    original = tmp_path / "fundamental-pointer.json"
    original.write_bytes(b"new generation")
    with journal.locked():
        again = publish_research_input(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
    assert again["research_request_ref"] == first["research_request_ref"]
    request = json.loads((tmp_path / first["research_request_ref"]["path"]).read_text())
    ref = request["company_evidence"]["fundamental_source"]["pointer"]
    assert ref["path"] != "fundamental-pointer.json"
    assert (tmp_path / ref["path"]).read_bytes() != original.read_bytes()
