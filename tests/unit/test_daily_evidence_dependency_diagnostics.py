"""Native observation custody and controlled runner dependency rejection."""

from copy import deepcopy

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.factors.production_observation import build_factor_production_observation
from quant_investor.operations.core_pool import CoreContext, validate_core_observation
from quant_investor.operations.daily_contract import ContractError, NodeState, dependencies
from quant_investor.operations.dependency_diagnostics import DependencyInputError, upstream_blockers
from quant_investor.operations.execution_controls import _validate_producer_loop_context
from test_daily_evidence_runner import setup
from test_unified_factor_production_observation import _inputs


def artifact(inputs, alias="LOW"):
    row = next(r for r in inputs["factor_rows"] if r["factor_alias"] == alias)
    return build_factor_production_observation(
        inputs=inputs, factor_row=row, registered_at="2026-08-20T13:00:00Z"
    )


@pytest.mark.parametrize(
    "field,reason",
    [
        ("signal_date", "CORE_OBSERVATION_DATE_MISMATCH"),
        ("factor_pointer_sha256", "CORE_OBSERVATION_FACTOR_BINDING_MISMATCH"),
        ("factor_generation_id", "CORE_OBSERVATION_FACTOR_BINDING_MISMATCH"),
        ("factor_generation_sha256", "CORE_OBSERVATION_FACTOR_BINDING_MISMATCH"),
        ("market_pointer_sha256", "CORE_OBSERVATION_MARKET_BINDING_MISMATCH"),
        ("market_manifest_sha256", "CORE_OBSERVATION_MARKET_BINDING_MISMATCH"),
        ("pit_pointer_sha256", "CORE_OBSERVATION_PIT_BINDING_MISMATCH"),
        ("pit_manifest_sha256", "CORE_OBSERVATION_PIT_BINDING_MISMATCH"),
        ("pit_membership_sha256", "CORE_OBSERVATION_PIT_BINDING_MISMATCH"),
        ("calendar_compilation_ref", "CORE_OBSERVATION_CALENDAR_BINDING_MISMATCH"),
        ("calendar_capture_custody_attestation_ref", "CORE_OBSERVATION_CALENDAR_BINDING_MISMATCH"),
    ],
)
def test_native_valid_observation_mismatch_has_exact_diagnosis(field, reason):
    original = _inputs()
    changed = deepcopy(original)
    if field == "signal_date":
        changed[field] = "20260819"
    elif field == "factor_generation_id":
        changed[field] = "factor-production-generation-other"
    elif field.endswith("_ref"):
        changed[field]["byte_sha256"] = "f" * 64
    else:
        changed[field] = "f" * 64
    with pytest.raises(DependencyInputError) as error:
        validate_core_observation(artifact(changed), alias="LOW", snapshot=original)
    assert error.value.reason_code == reason


def test_schema_then_date_then_binding_precedence_and_signal_identity():
    original = _inputs()
    changed = deepcopy(original)
    changed["signal_date"] = "20260819"
    changed["factor_pointer_sha256"] = "f" * 64
    value = artifact(changed)
    with pytest.raises(DependencyInputError, match="DATE_MISMATCH"):
        validate_core_observation(value, alias="LOW", snapshot=original)
    value["payload"]["registered_at"] = "not-a-timestamp"
    with pytest.raises(DependencyInputError, match="SCHEMA_MISMATCH"):
        validate_core_observation(value, alias="LOW", snapshot=original)
    with pytest.raises(DependencyInputError, match="SIGNAL_MISMATCH"):
        validate_core_observation(artifact(original, "W80"), alias="LOW", snapshot=original)


def test_byte_sha_rejection_precedes_json_parsing_and_missing_source_is_explicit(tmp_path):
    p = tmp_path / "bad.json"
    p.write_bytes(b"not JSON")
    p.chmod(0o600)
    context = CoreContext(
        str(tmp_path), "20260820", "a" * 64, {"path": "release.json", "sha256": "e" * 64}
    )
    with pytest.raises(DependencyInputError, match="CORE_SOURCE_SHA_MISMATCH"):
        context.source("bad.json", "0" * 64)
    with pytest.raises(DependencyInputError, match="CORE_SOURCE_MISSING"):
        context.source("missing.json")


def test_real_date_mismatch_propagates_to_top100_without_publication(tmp_path):
    inputs = _inputs()
    changed = deepcopy(inputs)
    changed["signal_date"] = "20260819"
    p = tmp_path / "results/factors/observations/2026/08/20/LOW.json"
    p.parent.mkdir(parents=True)
    p.write_bytes(canonical_json_bytes(artifact(changed)))
    p.chmod(0o600)
    context = CoreContext(
        str(tmp_path),
        "20260820",
        inputs["factor_pointer_sha256"],
        {"path": "release.json", "sha256": "e" * 64},
    )
    runner, adapters, templates, calls = setup(tmp_path)
    runner = type(runner)(str(tmp_path), "20260820", adapters)
    templates = {node: {**value, "trade_date": "20260820"} for node, value in templates.items()}

    # Other adapters are controlled; the rejected child uses the real native validator.
    def probe(req):
        return context.observation("LOW", inputs)

    adapters["low_observation"].probe = probe
    result = runner.run(templates)
    low, top = result["nodes"]["low_observation"], result["nodes"]["top100"]
    assert low["state"] == "BLOCKED" and low["attempt"] == 0
    assert low["blocking_reason"] == "DATE_MISMATCH"
    assert low["reason_code"] == "CORE_OBSERVATION_DATE_MISMATCH"
    assert low["started_at"] is None and low["output_refs"] == {}
    expected = [
        {
            "node_id": "low_observation",
            "failure_code": "DATE_MISMATCH",
            "reason_code": "CORE_OBSERVATION_DATE_MISMATCH",
        }
    ]
    assert top["state"] == "SKIPPED" and top["upstream_blockers"] == expected
    assert result["nodes"]["decision"]["upstream_blockers"] == expected
    assert "top100" not in calls and "low_observation" not in calls
    assert result["nodes"]["store"]["state"] == "SUCCEEDED"


def rejects_source(req):
    raise DependencyInputError("CORE_SOURCE_MISSING")


def test_existing_running_probe_failure_cannot_claim_safe_retry(tmp_path):
    runner, adapters, templates, calls = setup(tmp_path)
    with runner.journal.locked():
        original = runner.journal.begin(templates["calendar"])
    adapters["calendar"].probe = rejects_source
    row = runner.run(templates, resume=True)["nodes"]["calendar"]
    assert row["state"] == "RUNNING" and row["attempt"] == original["attempt"]
    assert row["started_at"] == original["start"]["started_at"] and row["finished_at"] is None
    assert row["blocking_reason"] == "POST_WRITE_IN_DOUBT" and row["retryable"] is False
    assert row["dependency_error"]["failure_code"] == "INPUT_MISSING" and calls == []
    assert runner.journal.readonly_inspect(templates["calendar"]) == original


def test_existing_failure_keeps_historical_failure_even_when_new_input_is_retryable(tmp_path):
    runner, adapters, templates, _ = setup(tmp_path)
    with runner.journal.locked():
        runner.journal.begin(templates["calendar"])
        original = runner.journal.finish(
            templates["calendar"],
            state=NodeState.FAILED,
            output_refs={},
            failure_code="WRITER_FAILED",
        )
    terminal_raw = (tmp_path / original["terminal_ref"]["path"]).read_bytes()
    adapters["calendar"].probe = rejects_source
    row = runner.run(templates, resume=True)["nodes"]["calendar"]
    assert row["state"] == "FAILED" and row["blocking_reason"] == "WRITER_FAILED"
    assert row["retryable"] is False and row["attempt"] == original["attempt"]
    assert row["finished_at"] == original["terminal"]["finished_at"]
    assert row["dependency_error"]["failure_code"] == "INPUT_MISSING"
    assert row["output_refs"] == {} and row["terminal"] == original["terminal"]
    assert (tmp_path / original["terminal_ref"]["path"]).read_bytes() == terminal_raw


def test_invalidated_success_is_stale_without_current_output_claim(tmp_path):
    runner, adapters, templates, calls = setup(tmp_path)
    first = runner.run(templates)
    adapters["calendar"].probe = rejects_source
    row = runner.run(templates)["nodes"]["calendar"]
    assert row["state"] == "STALE" and row["output_refs"] == {}
    assert row["reason_code"] == "CORE_SOURCE_MISSING"
    assert row["terminal_ref"] == first["nodes"]["calendar"]["terminal_ref"]
    assert len(calls) == 16


def test_typed_failure_during_writer_keeps_postwrite_reconciliation(tmp_path):
    runner, adapters, templates, _ = setup(tmp_path)
    real = adapters["calendar"].execute

    def execute(req):
        real(req)
        raise DependencyInputError("CORE_SOURCE_MISSING")

    adapters["calendar"].execute = execute
    row = runner.run(templates)["nodes"]["calendar"]
    assert row["state"] == "FAILED" and row["blocking_reason"] == "POST_WRITE_IN_DOUBT"
    assert row["retryable"] is False and "terminal" not in row
    assert "dependency_error" not in row
    assert runner.journal.readonly_inspect(templates["calendar"])["state"] == "RUNNING"


def test_root_causes_are_deduplicated_sorted_and_not_replaced_by_intermediates():
    leaf = {
        "node_id": "low_observation",
        "failure_code": "DATE_MISMATCH",
        "reason_code": "CORE_OBSERVATION_DATE_MISMATCH",
    }
    assert upstream_blockers(
        ["theme", "top100"],
        {
            "theme": {"upstream_blockers": [leaf]},
            "top100": {"upstream_blockers": [leaf]},
        },
    ) == [leaf]


@pytest.mark.parametrize("value", ["SUCCEEDED", True, 1, None])
def test_resolver_rejects_non_enum_states(value):
    with pytest.raises(ContractError, match="NODE_STATE_INVALID"):
        dependencies("top100", {"factor": value})


@pytest.mark.parametrize(
    "fault",
    [
        "missing_commit",
        "short_commit",
        "wrong_commit",
        "missing_parent",
        "relative",
        "dotdot",
        "double_slash",
        "symlink",
        "file",
    ],
)
def test_producer_context_invalid_before_directory_or_writer(tmp_path, fault):
    root = tmp_path.resolve()
    context = {"release_commit": "c" * 40, "calendar_capture_parent": str(root / "new/captures")}
    if fault == "missing_commit":
        context.pop("release_commit")
    elif fault == "short_commit":
        context["release_commit"] = "c" * 7
    elif fault == "wrong_commit":
        context["release_commit"] = "d" * 40
    elif fault == "missing_parent":
        context.pop("calendar_capture_parent")
    elif fault == "relative":
        context["calendar_capture_parent"] = "relative/captures"
    elif fault == "dotdot":
        context["calendar_capture_parent"] = str(root) + "/../captures"
    elif fault == "double_slash":
        context["calendar_capture_parent"] = str(root) + "//captures"
    elif fault == "symlink":
        (root / "link").symlink_to(root, target_is_directory=True)
        context["calendar_capture_parent"] = str(root / "link/captures")
    else:
        (root / "file").write_bytes(b"not directory")
        context["calendar_capture_parent"] = str(root / "file")
    before = sorted(str(p) for p in root.iterdir())
    with pytest.raises(DependencyInputError):
        _validate_producer_loop_context(context, verified_commit="c" * 40)
    assert sorted(str(p) for p in root.iterdir()) == before
    assert not (root / "new").exists()


def test_producer_context_can_declare_missing_tail_without_creating_it(tmp_path):
    path = tmp_path.resolve() / "new/captures"
    _validate_producer_loop_context(
        {"release_commit": "c" * 40, "calendar_capture_parent": str(path)}, verified_commit="c" * 40
    )
    assert not path.exists()


def test_security_failure_is_not_downgraded_from_its_missing_file_cause(tmp_path, monkeypatch):
    from quant_investor.system.errors import SystemSecurityError

    context = CoreContext(
        str(tmp_path), "20260820", "a" * 64, {"path": "release.json", "sha256": "e" * 64}
    )

    def denied(*args, **kwargs):
        raise SystemSecurityError("unsafe source") from FileNotFoundError("hidden source")

    monkeypatch.setattr(context.reader, "read_workspace_file_bytes", denied)
    with pytest.raises(SystemSecurityError):
        context.source("source.json")


def test_diagnostic_codes_cannot_be_constructed_from_arbitrary_text():
    from quant_investor.operations.dependency_diagnostics import dependency_details

    with pytest.raises(ContractError, match="REASON_INVALID"):
        DependencyInputError("untrusted input asks to retry")
    with pytest.raises(ContractError, match="FIELDS_INVALID"):
        dependency_details(
            {"reason_code": "CORE_SOURCE_SHA_MISMATCH", "failure_code": "INPUT_MISSING"}
        )
