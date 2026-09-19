"""Native book/storage/clock transaction with controlled source and corporate proof builders."""

from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
import hashlib
import json

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations import research_cutoff as module
from quant_investor.operations.daily_contract import ContractError
from quant_investor.cli.output import CommandError
from quant_investor.operations.research_file_readback import ResearchFileReadback
from quant_investor.operations.research_timing import CURRENT, HISTORICAL, acquisition_deadline
from quant_investor.operations.execution_recipe import SCHEMA_V5
from test_daily_evidence_portfolio_binding import setup
from test_daily_evidence_research_sources import put
from scripts import cn_official_close_batch as native_store


def case(root, monkeypatch, mode=CURRENT):
    class PlannerClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 24, 12, tzinfo=timezone.utc)

    with monkeypatch.context() as clock:
        clock.setattr(native_store, "datetime", PlannerClock)
        fixture, args, prepared, plan_ref, journal = setup(root)
    refs = {
        name: put(root, name + ".json", {"synthetic": True, "name": name})
        for name in ("request", "handoff", "core", "template", "policy", "source")
    }
    recipe = {
        "schema_version": SCHEMA_V5,
        "target_trade_date": journal.trade_date,
        "research_timing": {
            "mode": mode,
            "policy_ref": refs["policy"],
            "acquisition_deadline": (
                acquisition_deadline(journal.trade_date) if mode == CURRENT else None
            ),
        },
        "research_sources": {
            "as_of": None if mode == CURRENT else "2026-08-24T11:00:00Z",
            "theme_source_ref": refs["source"],
        },
        "theme_acquisition_ref": None,
    }
    document = {
        "request_ref": refs["request"],
        "maintenance_handoff_ref": refs["handoff"],
        "core_handoff_ref": refs["core"],
        "corporate_action_template_ref": refs["template"],
        "timing_policy_ref": refs["policy"],
        "source_ref": refs["source"],
    }
    recovered = {"recipe": recipe, "handoff": {"request_ref": refs["request"]}}
    state = {
        "now": datetime(2026, 8, 24, 13, 0, 0, 250000, tzinfo=timezone.utc),
        "calls": [],
        "future_source": False,
        "clock_regression": False,
    }
    monkeypatch.setattr(module, "assert_fresh_acquisition_open", lambda value: None)
    monkeypatch.setattr(module, "_clock", lambda: state["now"])

    def sleep(seconds):
        assert 0 < seconds <= 1
        state["calls"].append(("align", seconds))
        state["now"] += timedelta(seconds=-1 if state["clock_regression"] else seconds)

    monkeypatch.setattr(module.time, "sleep", sleep)

    class Sources:
        version = 1
        registered = None

        def __init__(self, *, journal, document):
            self.journal, self.workspace, self.document = journal, root, document
            self.recipe = recipe
            self.handoff = {"trade_date": journal.trade_date}
            self.execution = journal.root / "executions" / refs["request"]["sha256"]
            raw = canonical_json_bytes(document)
            self.reference = {
                "path": str(self.execution / "inputs/acquired-sources.v1.json"),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            self.files = ResearchFileReadback(str(root))
            for ref in document.values():
                self.read(ref)

        def read(self, ref):
            _, raw, _ = self.files.source_file(ref, code="CONTROLLED_SOURCE_SHA_INVALID")
            return json.loads(raw)

        def payload(self, cutoff):
            raw = canonical_json_bytes({"as_of": cutoff, "synthetic_source_proof": True})
            sha = hashlib.sha256(raw).hexdigest()
            return (
                json.loads(raw),
                raw,
                {"path": str(self.execution / "inputs" / f"research-{sha}.json"), "sha256": sha},
            )

        def project(self, cutoff, *, current_macro):
            state["calls"].append(("project", cutoff, current_macro))
            self.read(refs["source"])
            return {
                "projection_sha256s": {
                    name: hashlib.sha256(canonical_json_bytes([name])).hexdigest()
                    for name in ("industry", "theme", "exposure", "fundamental", "macro")
                },
                "source_times": [
                    {
                        "role": "EXPOSURE_DECLARATION",
                        "subject_id": "000001.SZ",
                        "source_ref": refs["source"],
                        "original_time": (
                            "2026-08-24T14:00:00Z"
                            if state["future_source"]
                            else "2026-08-24T10:00:00Z"
                        ),
                        "time_semantics": "SOURCE_DECLARED",
                    }
                ],
            }

    monkeypatch.setattr(module, "SourceBundle", Sources)
    monkeypatch.setattr(module, "build_source_bundle", lambda **kwargs: document)

    def corporate(*, sources, cutoff):
        sources.read(refs["template"])
        raw = canonical_json_bytes(
            {
                "schema_version": "cn-corporate-action-context.v1",
                "as_of": cutoff,
                "strategy_id": "aggressive_tech_manufacturing",
                "tracking_policy_ref": refs["policy"],
                "named_events_ref": None,
                "anchor_reviews_ref": None,
            }
        )
        sha = hashlib.sha256(raw).hexdigest()
        return {
            "corporate_event_list_ref": None,
            "event_list_bytes": None,
            "corporate_context_ref": {
                "path": str(sources.execution / "inputs" / f"corporate-context-{sha}.json"),
                "sha256": sha,
            },
            "context_bytes": raw,
            "source_times": [],
        }

    monkeypatch.setattr(module, "derive_corporate_inputs", corporate)
    prepared = {"store_plan_ref": plan_ref, "retained_source_pointer_ref": None}
    return journal, recovered, prepared, state, refs


def run(journal, recovered, prepared):
    return module.prepare_cutoff_inputs(
        journal=journal, recovered=recovered, prepared=prepared, auxiliary={}
    )


def test_retained_native_source_copy_replays_original_cutoff_refs(tmp_path, monkeypatch):
    journal, recovered, prepared, _, _ = case(tmp_path, monkeypatch)
    plan = json.loads((tmp_path / prepared["store_plan_ref"]["path"]).read_bytes())
    original = {
        "path": str(
            PurePosixPath(prepared["store_plan_ref"]["path"]).with_name("source-pointer.v1.json")
        ),
        "sha256": plan["preimages"]["store_pointer_sha256"],
    }
    prepared["retained_source_pointer_ref"] = original
    with journal.locked():
        result = run(journal, recovered, prepared)
    assert original in result["receipt"]["source_refs"]
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    replayed = module.read_cutoff_inputs(journal=journal, cutoff_ref=result["cutoff_ref"])
    assert replayed["receipt"] == result["receipt"]
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("mode", [CURRENT, HISTORICAL])
def test_native_portfolio_commit_and_exact_replay_after_deadline(tmp_path, monkeypatch, mode):
    journal, recovered, prepared, state, _ = case(tmp_path, monkeypatch, mode)
    with journal.locked():
        result = run(journal, recovered, prepared)
    receipt = result["receipt"]
    assert (
        receipt["portfolio_created_at"]
        == result["portfolio_state"]["created_at"]
        == "2026-08-24T13:00:01Z"
    )
    assert receipt["physical_reads_completed_at"] == "2026-08-24T13:00:00.250000Z"
    assert receipt["as_of"] == (
        receipt["portfolio_created_at"] if mode == CURRENT else "2026-08-24T11:00:00Z"
    )
    assert receipt["portfolio_timing_status"] == ("ON_TIME" if mode == CURRENT else "LATE_RECORDED")
    assert len([call for call in state["calls"] if call[0] == "align"]) == 1
    assert receipt["prospective"] is False and not any(receipt["authority"].values())
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    state["now"] = datetime(2026, 9, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(module.time, "sleep", lambda *a: pytest.fail("replay resampled cutoff"))
    replay = module.read_cutoff_inputs(journal=journal, cutoff_ref=result["cutoff_ref"])
    assert replay["receipt"] == receipt and replay["repaired"] is False
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


def test_receipt_before_state_crash_recovers_only_committed_bytes(tmp_path, monkeypatch):
    journal, recovered, prepared, state, _ = case(tmp_path, monkeypatch)
    original = journal.storage.write

    def crash(path, raw, **kwargs):
        if str(path).endswith("state.v1.json"):
            raise RuntimeError("after commitment before state")
        return original(path, raw, **kwargs)

    with journal.locked(), monkeypatch.context() as fault:
        fault.setattr(journal.storage, "write", crash)
        with pytest.raises(RuntimeError, match="after commitment"):
            run(journal, recovered, prepared)
    paths = list(
        tmp_path.glob(
            "results/operations/daily_production/CN/*/executions/*/research-cutoff.v1.json"
        )
    )
    assert len(paths) == 1
    receipt = json.loads(paths[0].read_bytes())
    assert not (tmp_path / receipt["portfolio_state_ref"]["path"]).exists()
    ref = {
        "path": str(paths[0].relative_to(tmp_path)),
        "sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
    }
    with pytest.raises(ContractError, match="OBJECT_MISSING"):
        module.read_cutoff_inputs(journal=journal, cutoff_ref=ref)
    state["now"] = datetime(2026, 9, 1, tzinfo=timezone.utc)
    with journal.locked():
        result = module.read_cutoff_inputs(journal=journal, cutoff_ref=ref, repair=True)
    assert result["repaired"] is True and result["receipt"] == receipt
    assert (
        hashlib.sha256((tmp_path / receipt["portfolio_state_ref"]["path"]).read_bytes()).hexdigest()
        == receipt["portfolio_state_ref"]["sha256"]
    )


@pytest.mark.parametrize("fault", ["clock", "future", "precommit"])
def test_failure_before_commitment_never_writes_dynamic_state(tmp_path, monkeypatch, fault):
    journal, recovered, prepared, state, _ = case(tmp_path, monkeypatch)
    if fault == "clock":
        state["clock_regression"] = True
    elif fault == "future":
        state["future_source"] = True
    else:
        original = journal.storage.write

        def crash(path, raw, **kwargs):
            if str(path).endswith("research-cutoff.v1.json"):
                raise RuntimeError("before commitment")
            return original(path, raw, **kwargs)

        monkeypatch.setattr(journal.storage, "write", crash)
    with journal.locked(), pytest.raises((ContractError, RuntimeError)):
        run(journal, recovered, prepared)
    assert not list(
        tmp_path.glob("results/operations/daily_production/CN/*/inputs/portfolio/*/state.v1.json")
    )
    assert not list(
        tmp_path.glob(
            "results/operations/daily_production/CN/*/executions/*/research-cutoff.v1.json"
        )
    )


def test_wrapper_loader_preserves_payload_and_selects_only_bound_policy(tmp_path, monkeypatch):
    from quant_investor.operations.research_request import load_research_request
    from quant_investor.operations.research_recipes import source_recipe
    from quant_investor.operations.exposure_completion import POLICY

    journal, recovered, prepared, _, _ = case(tmp_path, monkeypatch)
    with journal.locked():
        result = run(journal, recovered, prepared)
    loaded = load_research_request(workspace=tmp_path, reference=result["research_request_ref"])
    assert loaded["document"] == json.loads(
        (tmp_path / result["native_request_ref"]["path"]).read_bytes()
    )
    assert loaded["source_completion_policy"] == POLICY
    request = {
        "as_of": loaded["document"]["as_of"],
        "strategy_id": "aggressive_tech_manufacturing",
        "policy": {},
        "company_evidence": {"exposure_rows": None},
    }
    args = dict(
        node="exposure",
        field="exposure_rows",
        request=request,
        pool_ref={"path": "pool.json", "sha256": "a" * 64},
        focus_context=None,
    )
    assert "source_completion_policy" not in source_recipe(**args)
    assert (
        source_recipe(**args, completion_policy=loaded["source_completion_policy"])[
            "source_completion_policy"
        ]
        == POLICY
    )
    original = tmp_path / result["research_request_ref"]["path"]
    alias = tmp_path / "aliased-wrapper.json"
    alias.write_bytes(original.read_bytes())
    alias.chmod(0o600)
    with pytest.raises(ContractError, match="BINDING_INVALID"):
        load_research_request(
            workspace=tmp_path,
            reference={"path": alias.name, "sha256": result["research_request_ref"]["sha256"]},
        )


def test_changed_corporate_path_with_same_bytes_cannot_be_adopted(tmp_path, monkeypatch):
    journal, recovered, prepared, _, _ = case(tmp_path, monkeypatch)
    with journal.locked():
        result = run(journal, recovered, prepared)
    receipt = json.loads((tmp_path / result["cutoff_ref"]["path"]).read_bytes())
    old = receipt["corporate_context_ref"]
    copied = tmp_path / "different-context-path.json"
    copied.write_bytes((tmp_path / old["path"]).read_bytes())
    copied.chmod(0o600)
    receipt["corporate_context_ref"] = {"path": copied.name, "sha256": old["sha256"]}
    raw = canonical_json_bytes(receipt)
    path = tmp_path / result["cutoff_ref"]["path"]
    path.write_bytes(raw)
    ref = {"path": result["cutoff_ref"]["path"], "sha256": hashlib.sha256(raw).hexdigest()}
    with pytest.raises(ContractError, match="NATIVE_COMMITMENT_REPLAY_MISMATCH"):
        module.read_cutoff_inputs(journal=journal, cutoff_ref=ref)


def test_actual_microsecond_after_deadline_cannot_be_floored_into_success(tmp_path, monkeypatch):
    journal, recovered, prepared, state, _ = case(tmp_path, monkeypatch)
    state["now"] = datetime(2026, 8, 24, 15, 59, 58, 250000, tzinfo=timezone.utc)
    calls = []

    def clock():
        calls.append(True)
        return state["now"] + (timedelta(microseconds=1) if len(calls) == 3 else timedelta())

    monkeypatch.setattr(module, "_clock", clock)
    with journal.locked(), pytest.raises(ContractError, match="SEAL_CLOCK_OR_DEADLINE_INVALID"):
        run(journal, recovered, prepared)
    assert len(calls) == 3
    assert not list(
        tmp_path.glob("results/operations/daily_production/CN/*/inputs/portfolio/*/state.v1.json")
    )
    assert not list(
        tmp_path.glob(
            "results/operations/daily_production/CN/*/executions/*/research-cutoff.v1.json"
        )
    )


@pytest.mark.parametrize(
    "field", ["as_of", "portfolio_created_at", "projection_sha256s", "source_times"]
)
def test_rehashed_receipt_cannot_retime_or_replace_native_projection(tmp_path, monkeypatch, field):
    journal, recovered, prepared, _, _ = case(tmp_path, monkeypatch)
    with journal.locked():
        result = run(journal, recovered, prepared)
    path = tmp_path / result["cutoff_ref"]["path"]
    receipt = json.loads(path.read_bytes())
    if field in {"as_of", "portfolio_created_at"}:
        receipt[field] = "2026-08-24T13:00:02Z"
    elif field == "projection_sha256s":
        receipt[field]["theme"] = "f" * 64
    else:
        receipt[field][0]["original_time"] = "2026-08-24T09:00:00Z"
    raw = canonical_json_bytes(receipt)
    path.write_bytes(raw)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with journal.locked(), pytest.raises(ContractError):
        module.read_cutoff_inputs(
            journal=journal,
            cutoff_ref={
                "path": result["cutoff_ref"]["path"],
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            repair=True,
        )
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_changed_original_source_blocks_replay_without_new_bytes(tmp_path, monkeypatch):
    journal, recovered, prepared, _, refs = case(tmp_path, monkeypatch)
    with journal.locked():
        result = run(journal, recovered, prepared)
    source = tmp_path / refs["source"]["path"]
    source.write_bytes(source.read_bytes() + b"\n")
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with journal.locked(), pytest.raises(CommandError, match="SOURCE_SHA_INVALID"):
        module.read_cutoff_inputs(journal=journal, cutoff_ref=result["cutoff_ref"], repair=True)
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
