"""Default evaluator admission; controlled sources never claim real-time OOS."""

import json
import hashlib
from copy import deepcopy

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.prospective_timing import classify_daily_evidence
from quant_investor.factors.production_observation import validate_factor_production_observation

from quant_investor.factors.production_outcomes import classify_seal
from test_production_outcome_diagnostics import _settlement_fixture


def _put(root, path, value):
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = destination.parent
    while parent != root:
        parent.chmod(0o700)
        parent = parent.parent
    raw = canonical_json_bytes(value)
    destination.write_bytes(raw)
    destination.chmod(0o600)
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}


def _daily_fixture(tmp_path, monkeypatch, *, fault=None, two_factors=False):
    """Native observation + native timing math; full business replay is controlled."""
    from quant_investor.operations import outcome_native_replay
    from test_daily_evidence_prospective_timing import context
    from test_unified_factor_production_observation import _inputs
    from quant_investor.factors.production_observation import (
        build_factor_production_observation,
        _observation_path,
    )

    fixture = _settlement_fixture(
        tmp_path,
        monkeypatch,
        seal_time="2026-08-20T06:59:00Z",
        registered_at=(
            "2026-08-25T06:00:00Z" if fault == "registration" else "2026-08-20T13:00:00Z"
        ),
    )
    module, store, path, original, market, args = fixture
    inputs = store.read_observation_history()[0]
    market["prices"]["B"]["20260821"] = "12"
    observations = {"LOW": {"path": str(path), "sha256": hashlib.sha256(original).hexdigest()}}
    if two_factors:
        row = dict(_inputs()["factor_rows"][1], symbol_count=2)
        inputs["factor_rows"].append(row)
        inputs["signal_values"][row["factor_id"]] = {"A": (-1.0).hex(), "B": (-2.0).hex()}
        artifact = build_factor_production_observation(
            inputs=inputs, factor_row=row, registered_at="2026-08-20T13:00:00Z"
        )
        other_path = _observation_path("20260820", "W80")
        raw = canonical_json_bytes(artifact)
        store.write_exact_once(other_path, raw)
        observations["W80"] = {"path": str(other_path), "sha256": hashlib.sha256(raw).hexdigest()}
    timing = json.loads(
        json.dumps(context())
        .replace("20260908", "20260820")
        .replace("2026-09-08", "2026-08-20")
        .replace("07:01:00Z", "13:30:00Z")
        .replace("07:00:00Z", "13:00:00Z")
        .replace("07:00:30Z", "13:00:30Z")
        .replace("07:00:50Z", "13:20:00Z")
    )
    policy_sha = hashlib.sha256(canonical_json_bytes(timing["policy"])).hexdigest()
    for ref in (
        timing["handoff"]["prospective_policy_ref"],
        timing["recipe"]["policy_refs"]["prospective"],
    ):
        ref["sha256"] = policy_sha
    core = timing["core_timing"]
    core["factor_pointer_ref"]["sha256"] = inputs["factor_pointer_sha256"]
    timing["handoff"]["factor_pointer_ref"]["sha256"] = inputs["factor_pointer_sha256"]
    core["generation_seal_upper_bound"] = inputs["seal_time_upper_bound"]
    for alias, ref in observations.items():
        observation = validate_factor_production_observation(store.read(ref["path"]).data)[
            "payload"
        ]
        node = "low_observation" if alias == "LOW" else "w80_observation"
        core["observation_registration"][alias] = {
            "registered_at": observation["registered_at"],
            "observation_ref": ref,
        }
        timing["node_custody"]["nodes"][node]["output_refs"] = {alias: ref}
    late_node = {"registration": "low_observation", "decision": "decision", "store": "store"}.get(
        fault
    )
    if late_node:
        row = timing["node_custody"]["nodes"][late_node]
        row["completed_at"] = row["first_verified_at"] = "2026-08-25T06:00:00Z"
        if late_node in core["node_custody"]:
            core["node_custody"][late_node]["completed_at"] = row["completed_at"]
    if fault == "policy":
        timing["handoff"]["sealed_at"] = "2026-08-25T06:00:00Z"
    timing["synthetic"] = fault == "synthetic"
    if fault == "historical":
        timing["handoff"]["schema_version"] = "cn-daily-maintenance-handoff.v3"
    classified = classify_daily_evidence(**timing)
    ledger = {
        "schema_version": "cn-daily-evidence-ledger.v1",
        "trade_date": "20260820",
        "core_timing": core,
        "node_custody": timing["node_custody"],
        **{key: classified[key] for key in ("classification", "prospective", "synthetic")},
        "recomputed": fault in {"historical", "synthetic"},
    }
    ledger_ref = _put(tmp_path, "results/prospective/CN/20260820/evidence-ledger.v1.json", ledger)
    completion = {
        "schema_version": "cn-daily-eod-completion.v2",
        "trade_date": "20260820",
        "prospective_ledger_ref": ledger_ref,
        "node_terminal_refs": {
            node: row["terminal_ref"] for node, row in timing["node_custody"]["nodes"].items()
        },
    }
    journal = DailyJournal(str(tmp_path), "20260820")
    completion_ref = _put(tmp_path, str(journal.root / "completion.v1.json"), completion)
    calls = []

    def replay(**kwargs):
        calls.append(kwargs)
        return {
            "trade_date": "20260820",
            "completion_ref": deepcopy(completion_ref),
            "ledger_ref": deepcopy(ledger_ref),
            "validation_scope": "FULL_NATIVE_EOD_AVAILABILITY",
            **{
                key: ledger[key]
                for key in ("classification", "prospective", "synthetic", "recomputed")
            },
            "native_identity": {
                "release_install_ref": {"path": "controlled-install.json", "sha256": "a" * 64},
                "final_commit": "b" * 40,
                "final_tree": "c" * 40,
                "installed_code_manifest_sha256": "d" * 64,
                "repository_root": str(tmp_path),
                "operation": "completion_replay",
            },
            "snapshot": None,
        }

    monkeypatch.setattr(outcome_native_replay, "replay_outcome_daily_evidence", replay)
    return fixture, ledger, ledger_ref, completion, completion_ref, calls


def test_eligible_metrics_are_grouped_by_factor_and_replayed_on_read(tmp_path, monkeypatch):
    from quant_investor.factors.production_outcome_admission import (
        read_oos_summary,
        read_daily_admission,
    )

    fixture, _, _, _, _, calls = _daily_fixture(tmp_path, monkeypatch, two_factors=True)
    module, store, _, _, _, args = fixture
    result = module.settle_production_observations(**args)
    oos = result["oos_evidence"]
    assert len(calls) == 1 and oos["eligible_observation_count"] == 2
    assert oos["eligible_signal_day_count"] == 1
    assert sorted(
        group["1"]["metrics"]["rank_ic"]["mean"] for group in oos["by_factor"].values()
    ) == ["-1", "1"]
    for group in oos["by_factor"].values():
        assert group["1"]["eligible_origin_count"] == 1
        assert group["1"]["metrics"]["rank_ic"]["sample_count"] == 1
        assert group["60"]["metrics"]["rank_ic"] == {
            "state": "UNAVAILABLE",
            "sample_count": 0,
            "mean": None,
        }
    assert read_oos_summary(store=store, summary_ref=result["inventory_ref"])["oos_evidence"] == oos
    assert len(calls) == 2
    first = oos["admissions"][0]
    assert (
        read_daily_admission(store=store, admission_ref=first["admission_ref"])["projection"][
            "state"
        ]
        == "ELIGIBLE"
    )
    assert len(calls) == 3


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("registration", "LATE_REGISTERED"),
        ("decision", "LATE_REGISTERED"),
        ("store", "LATE_REGISTERED"),
        ("policy", "LATE_REGISTERED"),
        ("synthetic", "SYNTHETIC_EVIDENCE"),
        ("historical", "RETROSPECTIVE_RECOMPUTE"),
    ],
)
def test_native_timing_exclusions_reach_the_actual_default_evaluator(
    tmp_path, monkeypatch, fault, reason
):
    fixture, _, _, _, _, calls = _daily_fixture(tmp_path, monkeypatch, fault=fault)
    module, store, path, original, market, args = fixture
    result = module.settle_production_observations(**args)
    assert result["component_sealed_by_close_count"] == 1
    assert result["oos_evidence"]["eligible_observation_count"] == 0
    assert result["oos_evidence"]["exclusions_by_reason"] == {reason: 1}
    assert result["horizon_states"]["1"] == {"EVALUATED": 1}
    assert store.read(path).data == original and len(calls) == 1


def test_default_settlement_without_daily_completion_keeps_raw_diagnostics_only(
    tmp_path, monkeypatch
):
    module, store, path, original, market, args = _settlement_fixture(tmp_path, monkeypatch)
    result = module.settle_production_observations(**args)
    assert result["schema_version"] == "factor-production-diagnostics.v2"
    assert result["horizon_states"]["1"] == {"EVALUATED": 1}
    assert result["component_classification_scope"] == "SIGNAL_SEAL_TIMING_ONLY"
    assert "original_close_prospective_count" not in result
    evidence = result["oos_evidence"]
    assert evidence["eligible_observation_count"] == 0
    assert evidence["eligible_signal_day_count"] == 0
    assert evidence["factor_admission"] is False
    assert evidence["exclusions_by_reason"] == {"DAILY_COMPLETION_MISSING": 1}
    assert store.read(path).data == original
    assert result["provider_calls"] is False
    assert result["effectiveness_state"] == "FACTOR_EFFECTIVENESS_INSUFFICIENT_EVIDENCE"
    again = module.settle_production_observations(**args)
    assert again["new_evaluation_count"] == 0
    assert again["oos_evidence"] == evidence


def test_public_settle_command_uses_daily_admission_without_new_flags(
    tmp_path, monkeypatch, capsys
):
    from quant_investor.cli.main import main

    _, _, _, _, _, args = _settlement_fixture(tmp_path, monkeypatch)
    main(
        [
            "factor",
            "production-settle",
            "--workspace-root",
            args["workspace_root"],
            "--calendar-receipt",
            args["calendar_receipt"],
            "--expected-calendar-sha256",
            args["expected_calendar_sha256"],
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["schema_version"] == "factor-production-diagnostics.v2"
    assert result["oos_evidence"]["eligible_observation_count"] == 0
    assert result["oos_evidence"]["exclusions_by_reason"] == {"DAILY_COMPLETION_MISSING": 1}


def test_original_component_seal_diagnostic_is_preserved_without_oos_authority():
    result = classify_seal(
        signal_date="20260904",
        seal_time_upper_bound="2026-09-04T06:59:00Z",
        registered_at="2026-09-06T02:00:00Z",
    )
    assert result["cohort"] == "ORIGINAL_CLOSE_COHORT"
    assert result["prospective_eligible"] is True
    assert result["authority"] == "NON_AUTHORIZING"


@pytest.mark.parametrize(
    "fault", ["day", "pointer", "registration", "observation", "custody", "output"]
)
def test_valid_day_cannot_admit_another_observation_binding(tmp_path, monkeypatch, fault):
    fixture, ledger, ledger_ref, completion, completion_ref, _ = _daily_fixture(
        tmp_path, monkeypatch
    )
    module, _, _, _, _, args = fixture
    core = ledger["core_timing"]
    if fault == "day":
        ledger["trade_date"] = "20260821"
    elif fault == "pointer":
        core["factor_pointer_ref"]["sha256"] = "f" * 64
    elif fault == "registration":
        core["observation_registration"]["LOW"]["registered_at"] = "2026-08-20T12:00:00Z"
    elif fault == "observation":
        core["observation_registration"]["LOW"]["observation_ref"]["path"] = "copied.json"
    elif fault == "custody":
        core["node_custody"]["low_observation"]["terminal_ref"]["sha256"] = "f" * 64
    else:
        ledger["node_custody"]["nodes"]["low_observation"]["output_refs"]["LOW"]["sha256"] = (
            "f" * 64
        )
    ledger_ref.update(_put(tmp_path, ledger_ref["path"], ledger))
    completion_ref.update(_put(tmp_path, completion_ref["path"], completion))
    result = module.settle_production_observations(**args)["oos_evidence"]
    assert result["eligible_observation_count"] == 0
    assert result["exclusions_by_reason"] == {"DAILY_OBSERVATION_BINDING_MISMATCH": 1}


def test_later_ledger_changes_admission_without_rewriting_raw_outcome(tmp_path, monkeypatch):
    fixture, _, _, _, completion_ref, calls = _daily_fixture(tmp_path, monkeypatch)
    module, store, _, _, _, args = fixture
    path = tmp_path / completion_ref["path"]
    raw = path.read_bytes()
    path.unlink()
    first = module.settle_production_observations(**args)
    assert first["oos_evidence"]["eligible_observation_count"] == 0 and calls == []
    originals = {
        row["evaluation_ref"]["path"]: (
            (tmp_path / row["evaluation_ref"]["path"]).read_bytes(),
            (tmp_path / row["evaluation_ref"]["path"]).stat().st_mtime_ns,
        )
        for row in first["outcomes"]
        if row["state"] == "EVALUATED"
    }
    path.write_bytes(raw)
    path.chmod(0o600)
    second = module.settle_production_observations(**args)
    assert second["new_evaluation_count"] == 0
    assert second["oos_evidence"]["eligible_observation_count"] == 1 and len(calls) == 1
    for name, (before, mtime) in originals.items():
        assert (tmp_path / name).read_bytes() == before
        assert (tmp_path / name).stat().st_mtime_ns == mtime


def test_absent_to_present_during_batch_is_changed_without_replaying_new_bytes(
    tmp_path, monkeypatch
):
    from quant_investor.factors import production_outcome_admission as admission

    fixture, _, _, _, completion_ref, calls = _daily_fixture(tmp_path, monkeypatch)
    module, _, _, _, _, args = fixture
    path = tmp_path / completion_ref["path"]
    raw = path.read_bytes()
    path.unlink()
    original = admission.persist_source
    inserted = []

    def publish(store, value):
        ref = original(store, value)
        if value.get("schema_version") == admission.ADMISSION_SCHEMA and not inserted:
            path.write_bytes(raw)
            path.chmod(0o600)
            inserted.append(True)
        return ref

    monkeypatch.setattr(admission, "persist_source", publish)
    oos = module.settle_production_observations(**args)["oos_evidence"]
    assert oos["exclusions_by_reason"] == {"DAILY_EVIDENCE_CHANGED": 1}
    assert oos["eligible_observation_count"] == 0 and calls == []


def test_post_summary_drift_publishes_invalid_correction_and_old_positive_is_untrusted(
    tmp_path, monkeypatch
):
    from quant_investor.factors import production_outcome_admission as admission

    fixture, _, _, _, completion_ref, calls = _daily_fixture(tmp_path, monkeypatch)
    module, store, _, _, _, args = fixture
    original = admission.persist_source
    summaries = []

    def publish(selected_store, value):
        ref = original(selected_store, value)
        if value.get("schema_version") == "factor-production-diagnostics.v2":
            summaries.append(ref)
            if len(summaries) == 1:
                (tmp_path / completion_ref["path"]).write_bytes(b"changed")
        return ref

    monkeypatch.setattr(admission, "persist_source", publish)
    result = module.settle_production_observations(**args)
    assert len(summaries) == 2 and result["inventory_ref"] == summaries[1]
    assert result["oos_evidence"]["exclusions_by_reason"] == {"DAILY_EVIDENCE_CHANGED": 1}
    assert len(calls) == 1
    with pytest.raises(ValueError):
        admission.read_oos_summary(store=store, summary_ref=summaries[0])


def test_original_observation_drift_invalidates_both_factors_full_day_proof(tmp_path, monkeypatch):
    from quant_investor.factors import production_outcome_admission as admission

    fixture, _, _, _, _, _ = _daily_fixture(tmp_path, monkeypatch, two_factors=True)
    module, store, path, _, _, args = fixture
    original = admission.persist_source
    changed = []

    def publish(selected_store, value):
        ref = original(selected_store, value)
        if value.get("schema_version") == "factor-production-diagnostics.v2" and not changed:
            (tmp_path / path).write_bytes(b"changed")
            changed.append(True)
        return ref

    monkeypatch.setattr(admission, "persist_source", publish)
    result = module.settle_production_observations(**args)["oos_evidence"]
    assert result["eligible_observation_count"] == 0
    assert result["exclusions_by_reason"] == {"DAILY_EVIDENCE_CHANGED": 2}


def test_second_new_drift_aborts_summary_and_cursor_update(tmp_path, monkeypatch):
    from quant_investor.factors import production_outcome_admission as admission

    fixture, _, _, _, _, _ = _daily_fixture(tmp_path, monkeypatch, two_factors=True)
    module, _, _, _, _, args = fixture
    original = admission.persist_source
    summaries = []

    def publish(store, value):
        ref = original(store, value)
        if value.get("schema_version") == "factor-production-diagnostics.v2":
            outcomes = [row for row in value["outcomes"] if row["state"] == "EVALUATED"]
            target = outcomes[len(summaries)]["evaluation_ref"]["path"]
            (tmp_path / target).write_bytes(b"changed")
            summaries.append(ref)
        return ref

    monkeypatch.setattr(admission, "persist_source", publish)
    with pytest.raises(ValueError, match="OUTCOME_DAILY_EVIDENCE_UNSTABLE"):
        module.settle_production_observations(**args)
    assert len(summaries) == 2
    assert not (tmp_path / "results/factors/outcome-processing.json").exists()


@pytest.mark.parametrize(
    "fault", ["metrics", "observation", "origin", "horizon", "factor", "duplicate"]
)
def test_aggregate_uses_immutable_artifacts_and_exact_bindings(tmp_path, monkeypatch, fault):
    from quant_investor.factors import production_outcome_admission as admission

    fixture, _, _, _, _, _ = _daily_fixture(tmp_path, monkeypatch)
    module, store, _, _, _, args = fixture
    result = module.settle_production_observations(**args)
    row = deepcopy(next(r for r in result["outcomes"] if r["state"] == "EVALUATED"))
    observation = validate_factor_production_observation(
        store.read(row["observation_ref"]["path"]).data
    )["payload"]
    batch = admission.AdmissionBatch(
        store, [{"observation": observation, "observation_ref": row["observation_ref"]}]
    )
    if fault == "metrics":
        row["metrics"]["ic"] = {"state": "AVAILABLE", "value": "-999"}
        value = batch.evidence([row], publish=False)
        assert value["by_factor"][row["factor_id"]]["1"]["metrics"]["ic"]["mean"] == "1"
        return
    if fault == "observation":
        row["observation_ref"]["sha256"] = "f" * 64
    elif fault == "origin":
        row["origin_session"] = "20260821"
    elif fault == "horizon":
        row["horizon"] = 5
    elif fault == "factor":
        row["factor_id"] = "another-factor"
    with pytest.raises(ValueError):
        batch.evidence([row, row] if fault == "duplicate" else [row], publish=False)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "not numeric", True, 1, None])
def test_invalid_available_metric_never_becomes_an_oos_sample(value):
    from quant_investor.factors.production_outcome_admission import _metric_value

    with pytest.raises(ValueError, match="OUTCOME_OOS_METRIC_INVALID"):
        _metric_value({"state": "AVAILABLE", "value": value})
