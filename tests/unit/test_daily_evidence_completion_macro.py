"""Binding-only tests; native integration proof is produced by the installed DAG."""

import copy
import hashlib
from quant_investor.contracts import canonical_json_bytes
import pytest
from quant_investor.operations import completion_macro as replay
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import FALSE_AUTHORITY, request_identity
from quant_investor.intelligence.daily_evidence import build_market_risk_evidence
from test_daily_evidence_completion_readback import put
from test_daily_evidence_dag_journal import request


def fixture(root, monkeypatch, change, *, freshness=False):
    cutoff = "2026-09-04T13:30:00Z"
    closure = {"target_date": "20260904", "available_at": "2026-09-04T13:10:00.000001Z"}
    pool_ref = put(root, "pool-ref.json", {"fixture": "pool binding only"})
    if freshness:
        from test_macro_observation_evidence_store import _publish_with_evidence, _row

        observation_root = root / "data/parquet/cn/macro_observations"
        _publish_with_evidence(
            observation_root,
            _row(
                period="2026-08-31",
                available="2026-09-01T01:00:00Z",
            ),
            run_id="g1",
        )
        pointer = observation_root / "_latest.json"
        frozen = root / "frozen-pointer.json"
        frozen.write_bytes(pointer.read_bytes())
        frozen.chmod(0o600)
        pointer.unlink()
        closure["frozen_pointers"] = {
            "observations": {
                "current_path": str(pointer.relative_to(root)),
                "generation_id": "g1",
                "frozen_ref": {
                    "path": "frozen-pointer.json",
                    "sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
                },
            }
        }
    if change == "late":
        closure["available_at"] = "2026-09-04T13:30:00.000001Z"
    digest = hashlib.sha256(canonical_json_bytes(closure)).hexdigest()
    closure_ref = put(root, f"results/intelligence/macro_readiness/20260904/{digest}.json", closure)
    risk = {"classification": "CANONICAL_MACRO_READY", "source": closure_ref}
    original_value = {"as_of": cutoff, "company_evidence": {"macro_risk": risk}}
    recipe = {"as_of": cutoff, "macro_risk": copy.deepcopy(risk)}
    if freshness:
        original_value.update(
            strategy_id="aggressive_tech_manufacturing", policy={"fixture": "exact recipe binding"}
        )
        recipe = replay.source_recipe(
            node="macro",
            field="macro_risk",
            request=original_value,
            pool_ref=pool_ref,
            focus_context=None,
        )
    original = put(root, "original.json", original_value)
    if change == "selector":
        recipe["freshness_contract"] = "unknown"
    if change == "source":
        recipe["macro_risk"]["source"]["sha256"] = "f" * 64
    recipe_ref = put(root, "recipe.json", recipe)
    artifact = build_market_risk_evidence(
        source_path=closure_ref["path"],
        source_sha256=closure_ref["sha256"],
        blocker_codes=[],
        classification="CANONICAL_MACRO_READY",
        as_of=cutoff,
    )
    artifact_ref = put(root, "artifact.json", artifact)
    freshness_ref = None
    if freshness:
        from functools import partial
        from quant_investor.cli.unified import _daily_source_file

        report = replay.macro_freshness_from_closure(
            workspace=root,
            as_of=cutoff,
            closure_ref=closure_ref,
            source_file=partial(_daily_source_file, root),
            closure=closure,
        )
        if change == "freshness_state":
            from quant_investor.contracts import seal_artifact

            report["payload"]["freshness_state"] = "FRESH"
            report = seal_artifact(replay.FRESHNESS_KIND, report["payload"], created_at=cutoff)
        freshness_ref = put(root, "freshness.json", report)
    result = {"as_of": cutoff, "artifacts": [artifact]}
    if change == "cutoff":
        result["as_of"] = "2026-09-04T13:31:00Z"
    result_ref = put(root, "result.json", result)
    dc = {
        "result_ref": result_ref,
        "research_cutoff": cutoff,
        "captured_at": "2026-09-04T14:00:00Z",
    }
    dc_ref = put(root, "decision-capture.json", dc)
    refs = {}
    for name, inputs in [
        ("macro", {"recipe": recipe_ref, **({"pool": pool_ref} if freshness else {})}),
        ("decision", {"native_request": original}),
    ]:
        req = {**request(), "node_id": name, "input_refs": inputs}
        key = request_identity(req)[1]
        base = f"nodes/{name}/{key}"
        put(root, base + "/request.json", req)
        if name == "macro":
            cap = {
                "schema_version": "cn-daily-research-node-capture.v1",
                "state": "SUCCEEDED",
                "request_key": key,
                "artifact_refs": [artifact_ref] + ([freshness_ref] if freshness else []),
                "authority": FALSE_AUTHORITY,
                "research_cutoff": cutoff,
                "captured_at": "2026-09-04T14:00:00Z",
            }
            if change == "custody":
                cap["captured_at"] = "2026-09-04T14:00:02Z"
            outputs = {"capture": put(root, "macro-capture.json", cap), "artifact": artifact_ref}
            if freshness:
                outputs[replay.FRESHNESS_KIND] = freshness_ref
            if change == "freshness_ref":
                outputs["artifact_1"] = outputs.pop(replay.FRESHNESS_KIND)
        else:
            outputs = {"capture": dc_ref, "result": result_ref}
        refs[name] = put(
            root,
            base + "/attempt-0001/terminal.json",
            {
                "request_key": key,
                "output_refs": outputs,
                "finished_at": "2026-09-04T14:00:01Z",
            },
        )
    monkeypatch.setattr(
        replay,
        "inspect_recorded_completion",
        lambda **kw: {
            "recorded_completion": {
                "node_terminal_refs": refs,
                "native_inputs_ref": put(root, "inputs.json", {"research_request_ref": original}),
                "synthetic": True,
            }
        },
    )
    monkeypatch.setattr(replay.ResearchCapture, "read", lambda *args: (dc, result))
    monkeypatch.setattr(replay, "validate_macro_readiness_closure", lambda **kw: kw["closure"])


@pytest.mark.parametrize(
    "change,error",
    [
        ("source", "CAPTURE_BINDING"),
        ("cutoff", "CUTOFF_MISMATCH"),
        ("custody", "CUSTODY_INVALID"),
        ("late", "UNAVAILABLE_AT_ORIGINAL"),
    ],
)
def test_completed_macro_rejects_broken_original_binding(tmp_path, monkeypatch, change, error):
    fixture(tmp_path, monkeypatch, change)
    with pytest.raises(ContractError, match=error):
        replay.replay_completed_macro(
            workspace=str(tmp_path), trade_date="20260904", completion_ref={}
        )


def test_completed_macro_fractional_time_and_no_admission(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch, None)
    # Prime fixture-only recorded completion so its test writer is outside inventory.
    recorded = replay.inspect_recorded_completion()
    monkeypatch.setattr(replay, "inspect_recorded_completion", lambda **kw: recorded)

    def inventory():
        return {
            str(p): (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None)
            for p in tmp_path.rglob("*")
        }

    before = inventory()
    result = replay.replay_completed_macro(
        workspace=str(tmp_path), trade_date="20260904", completion_ref={}
    )
    assert result["consumer_admission"] is False
    assert result["synthetic"] is True
    assert before == inventory()


def test_missing_completion_cannot_enter_macro_replay(tmp_path):
    with pytest.raises(ContractError):
        replay.replay_completed_macro(
            workspace=str(tmp_path), trade_date="20260904", completion_ref={}
        )


@pytest.mark.parametrize(
    "change,error",
    [
        (None, None),
        ("freshness_state", "FRESHNESS_DOES_NOT_REPLAY"),
        ("freshness_ref", "CAPTURE_BINDING"),
        ("selector", "CONTRACT_INVALID"),
    ],
)
def test_new_macro_recipe_replays_native_freshness_without_current_head(
    tmp_path, monkeypatch, change, error
):
    from quant_investor.macro import store

    fixture(tmp_path, monkeypatch, change, freshness=True)
    recorded = replay.inspect_recorded_completion()
    monkeypatch.setattr(replay, "inspect_recorded_completion", lambda **kw: recorded)

    def forbidden(*args, **kwargs):
        pytest.fail("completed Macro read current pointer")

    monkeypatch.setattr(store, "_optional_pointer_bytes", forbidden)
    before = {
        str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file()
    }
    kwargs = dict(workspace=str(tmp_path), trade_date="20260904", completion_ref={})
    if error:
        with pytest.raises(Exception, match=error):
            replay.replay_completed_macro(**kwargs)
    else:
        result = replay.replay_completed_macro(**kwargs)
        assert result["macro_artifact"]["kind"] == "market_risk_evidence"
        assert result["consumer_admission"] is False
    assert before == {
        str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file()
    }
