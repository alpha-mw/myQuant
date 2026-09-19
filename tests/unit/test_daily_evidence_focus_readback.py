"""Completed-source ref mapping with a controlled deterministic projector.

Native projection/adapter execution is covered in focus_sources; this isolates
the completed-reader boundary, including the new literal output names.
"""

from datetime import datetime, timezone
from copy import deepcopy

import pytest

from quant_investor.operations import completion_research as replay
from quant_investor.operations.research_projection import Projected
from quant_investor.operations.research_recipes import source_recipe
from quant_investor.operations.daily_contract import NodeState, ContractError
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY
from quant_investor.intelligence.storage import (
    DailyResearchPoolStore,
    approved_theme_policy_v2,
    publish_theme_policy_v2,
)
from quant_investor.intelligence.theme_sources import THEME_SOURCE_V2
from quant_investor.intelligence.low_frequency import FRESHNESS_KIND, FRESHNESS_CONTRACT
from test_unified_daily_intelligence_storage import _pool_rank
from test_daily_evidence_research_sources import put
from test_daily_evidence_focus_sources import pit_context, inventory
from test_daily_evidence_dag_journal import request


@pytest.mark.parametrize("freshness", [False, True])
@pytest.mark.parametrize(
    "fault",
    [None, "named_ref", "artifact_bytes", "recipe", "freshness_ref", "freshness_bytes", "selector"],
)
def test_completed_reader_preserves_focus_ref_contract(tmp_path, monkeypatch, fault, freshness):
    if not freshness and fault in {"freshness_ref", "freshness_bytes"}:
        pytest.skip("freshness ref exists only in versioned recipe")
    policy = approved_theme_policy_v2()
    published = publish_theme_policy_v2(tmp_path)
    rank = _pool_rank(tmp_path, policy, signal_date="20260828")
    pool = DailyResearchPoolStore(tmp_path).publish(
        rank=rank,
        expected_policy_sha256=published["daily_policy_sha256"],
        policy_path=published["daily_policy_path"],
        before_publish=lambda: None,
    )
    pool_ref = {"path": pool["manifest_path"], "sha256": pool["manifest_sha256"]}
    source = put(tmp_path, "source.json", {"fixture": "descriptor binding only"})
    descriptor = {
        "dc_plan": source,
        "dc_capture": source,
        "dc_partitions": [source],
        "tdx_plan": None,
        "tdx_capture": None,
        "tdx_partitions": [],
    }
    original = {
        "as_of": "2026-08-28T13:30:00Z",
        "strategy_id": "aggressive_tech_manufacturing",
        "policy": policy,
        "theme_source": {
            "schema_version": THEME_SOURCE_V2,
            "pool": descriptor,
            "pcb_ai_hardware": None,
        },
        "industry_source": None,
        "company_evidence": {"exposure_rows": [], "fundamental_source": None},
        "expected_factor_pointer_sha256": rank["payload"]["factor_pointer_sha256"],
    }
    original_ref = put(tmp_path, "original-request.json", original)
    focus = pit_context(tmp_path)
    monkeypatch.setattr(replay, "derive_focus_context", lambda **kwargs: deepcopy(focus))
    artifacts = {
        "theme": [
            {"kind": "ordinary-theme-fixture"},
            {"kind": "pcb_ai_hardware_membership", "fixture": True},
        ],
        "industry": [{"kind": "ordinary-industry-fixture"}],
        "exposure": [
            {"kind": "ordinary-exposure-fixture"},
            {"kind": "pcb_ai_hardware_evidence", "fixture": True},
        ],
        "fundamental": [{"kind": "ordinary-fundamental-fixture"}],
    }
    names = {
        "theme": ["artifact", "pcb_ai_hardware_membership"],
        "exposure": ["artifact", "pcb_ai_hardware_evidence"],
        "industry": ["artifact"],
        "fundamental": ["artifact"],
    }
    if freshness:
        artifacts["fundamental"].append(
            {"kind": FRESHNESS_KIND, "fixture": "deterministic projector only"}
        )
        names["fundamental"].append(FRESHNESS_KIND)
    monkeypatch.setattr(
        replay,
        "project_research_source",
        lambda **kwargs: Projected(deepcopy(artifacts[kwargs["node"]]), NodeState.SUCCEEDED),
    )
    journal = DailyJournal(str(tmp_path), "20260828")
    terminals = {
        "top100": put(tmp_path, "top-terminal.json", {"output_refs": {"manifest.json": pool_ref}})
    }
    with journal.locked():
        for node, field in [
            ("theme", "theme_source"),
            ("industry", "industry_source"),
            ("exposure", "exposure_rows"),
            ("fundamental", "fundamental_source"),
        ]:
            recipe = source_recipe(
                node=node,
                field=field,
                request=original,
                pool_ref=pool_ref,
                focus_context=focus,
                freshness_contract=FRESHNESS_CONTRACT if freshness else None,
            )
            if fault == "selector" and node == "fundamental":
                recipe["freshness_contract"] = "unknown"
            recipe_ref = put(tmp_path, f"recipes/{node}.json", recipe)
            req = {
                **request(),
                "trade_date": "20260828",
                "node_id": node,
                "input_refs": {"pool": pool_ref, "recipe": recipe_ref},
            }
            started = journal.begin(req)
            refs = [
                put(tmp_path, f"source-artifacts/{node}-{index}.json", value)
                for index, value in enumerate(artifacts[node])
            ]
            cap = put(
                tmp_path,
                f"captures/{node}.json",
                {
                    "schema_version": "cn-daily-research-node-capture.v1",
                    "request_key": started["request_key"],
                    "state": "SUCCEEDED",
                    "artifact_refs": refs,
                    "authority": FALSE_AUTHORITY,
                    "research_cutoff": original["as_of"],
                    "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            )
            output = {**dict(zip(names[node], refs)), "capture": cap}
            terminals[node] = journal.finish(req, state=NodeState.SUCCEEDED, output_refs=output)[
                "terminal_ref"
            ]
    record = {
        "node_terminal_refs": terminals,
        "native_inputs_ref": put(
            tmp_path, "native-inputs.json", {"research_request_ref": original_ref}
        ),
    }
    monkeypatch.setattr(
        replay,
        "inspect_recorded_completion",
        lambda **kwargs: {"recorded_completion": deepcopy(record)},
    )
    if fault == "freshness_ref":
        import json

        ref = terminals["fundamental"]
        value = json.loads((tmp_path / ref["path"]).read_bytes())
        value["output_refs"]["artifact_1"] = value["output_refs"].pop(FRESHNESS_KIND)
        terminals["fundamental"] = put(tmp_path, ref["path"], value)
    elif fault == "freshness_bytes":
        (tmp_path / "source-artifacts/fundamental-1.json").write_bytes(b"changed")
    elif fault == "named_ref":
        import json

        ref = terminals["exposure"]
        value = json.loads((tmp_path / ref["path"]).read_bytes())
        value["output_refs"]["artifact_1"] = value["output_refs"].pop("pcb_ai_hardware_evidence")
        terminals["exposure"] = put(tmp_path, ref["path"], value)
    elif fault == "artifact_bytes":
        (tmp_path / "source-artifacts/exposure-1.json").write_bytes(b"changed")
    elif fault == "recipe":
        (tmp_path / "recipes/exposure.json").write_bytes(b"changed")
    before = inventory(tmp_path)
    kwargs = dict(workspace=str(tmp_path), trade_date="20260828", completion_ref=source)
    if fault:
        with pytest.raises((ContractError, ValueError)):
            replay.replay_completed_research_sources(**kwargs)
    else:
        result = replay.replay_completed_research_sources(**kwargs)
        assert result["artifacts"] == artifacts and result["consumer_admission"] is False
    assert before == inventory(tmp_path)
